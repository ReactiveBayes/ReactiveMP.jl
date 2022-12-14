export AddonLogScale, getlogscale

using Distributions

import Base: prod, string

struct AddonLogScale{T} <: AbstractAddon
    logscale::T
end

AddonLogScale() = AddonLogScale(nothing)

struct AddonProdLogScale <: AbstractAddonProd end

getlogscale(addon::AddonLogScale) = addon.logscale
getlogscale(addons::NTuple{N, AbstractAddon}) where {N} = mapreduce(getlogscale, +, filter(addon -> addon isa AddonLogScale, addons))

# TODO: for later review, do we need such a fallback
# getlogscale(::AbstractAddon) = 0

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