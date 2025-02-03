export AddonLogScale, getlogscale

using Distributions

import Base: prod, string

struct AddonLogScale{T} <: AbstractAddon
    logscale::T
end

AddonLogScale() = AddonLogScale(nothing)

getlogscale(addon::AddonLogScale) = addon.logscale
getlogscale(::Nothing) = error("Log-scale addon is not available. Make sure to include AddonLogScale in the addons. Currently, log scale factors are only supported for very specific nodes and messages in sum-product updates. Extensions to variational message passing are not yet supported.")

function getlogscale(addons::NTuple{N, AbstractAddon}) where {N}
    logscales = filter(addon -> addon isa AddonLogScale, addons)
    if length(logscales) === 0
        error("Log-scale addon is not available. Make sure to include AddonLogScale in the addons.")
    end
    return mapreduce(getlogscale, +, logscales)
end

function message_mapping_addon(::AddonLogScale{Nothing}, mapping, messages, marginals, result::Distribution)
    # Here we assume
    # 1. If log-scale value has not been computed during the message update rule
    # 2. Either all messages or marginals are of type PointMass
    # 3. The result of the message update rule is a proper distribution
    #  THEN: logscale is equal to zero
    #  OTHERWISE: show an error
    #  This logic probably can be improved, e.g. if some tracks conjugacy between the node and messages
    if isnothing(marginals) && all(data -> data isa PointMass, messages)
        return AddonLogScale(0)
    elseif isnothing(messages) && all(data -> data isa PointMass, marginals)
        return AddonLogScale(0)
    else
        error("Log-scale value has not been computed for the message update rule = $(mapping)")
    end
end

# Log scale macro for the message update rules
macro logscale(lambda)
    @capture(lambda, (body_)) || error("Error in macro. Lambda body specification is incorrect")
    # return expression for @logscale
    return esc(:(ReactiveMP.@invokeaddon AddonLogScale $body))
end

function multiply_addons(left_addon::AddonLogScale{Missing}, right_addon::AddonLogScale, new_dist, left_dist::Missing, right_dist)
    return right_addon
end

function multiply_addons(left_addon::AddonLogScale, right_addon::AddonLogScale{Missing}, new_dist, left_dist, right_dist::Missing)
    return left_addon
end

function multiply_addons(left_addon::AddonLogScale, right_addon::AddonLogScale, new_dist, left_dist, right_dist)

    # fetch log scales from addons
    left_logscale = getlogscale(left_addon)
    right_logscale = getlogscale(right_addon)

    # compute new logscale
    new_logscale = compute_logscale(new_dist, left_dist, right_dist)

    # return updated logscale addon
    return AddonLogScale(left_logscale + right_logscale + new_logscale)
end

function string(addon::AddonLogScale)
    return string("log-scale = ", getlogscale(addon), "; ")
end
