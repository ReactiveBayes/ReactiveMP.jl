export AddonLogScale, getlogscale

using Distributions

import Base: prod, string

struct AddonLogScale{T} <: AbstractAddon
    logscale::T
end

AddonLogScale() = AddonLogScale(nothing)

struct AddonProdLogScale <: AbstractAddonProd end

getlogscale(addon::AddonLogScale) = addon.logscale

function getlogscale(addons::NTuple{N, AbstractAddon}) where {N}
    logscales = filter(addon -> addon isa AddonLogScale, addons)
    if length(logscales) === 0
        error("Log-scale addon is not available.")
    end
    return mapreduce(getlogscale, +, logscales)
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
    new_logscale = prod(AddonProdLogScale(), new_dist, left_dist, right_dist)

    # return updated logscale addon
    return AddonLogScale(left_logscale + right_logscale + new_logscale)
end

function string(addon::AddonLogScale)
    return string("log-scale = ", getlogscale(addon), "; ")
end
