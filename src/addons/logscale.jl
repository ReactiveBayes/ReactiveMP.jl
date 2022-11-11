export AddonLogScale, getlogscale

using Distributions

import Base: prod, string

struct AddonLogScale{T} <: AbstractAddon
    logscale::T
end
AddonLogScale() = AddonLogScale(nothing)
struct AddonProdLogScale <: AbstractAddonProd end

getlogscale(addon::AddonLogScale) = addon.logscale
getlogscale(::AbstractAddon) = 0
getlogscale(addons::NTuple{N, AbstractAddon}) where {N} = mapreduce(getlogscale, +, addons)

function multiply_addons(left_addon::AddonLogScale, right_addon::AddonLogScale, new_dist::Distribution, left_message::Message, right_message::Message)

    # fetch addons and messages
    left_logscale = getlogscale(left_addon)
    right_logscale = getlogscale(right_addon)
    left_dist = getdata(left_message)
    right_dist = getdata(right_message)

    # compute new logscale
    new_logscale = prod(AddonProdLogScale(), new_dist, left_dist, right_dist)

    # return updated logscale addon
    return AddonLogScale(left_logscale + right_logscale + new_logscale)
end

function string(addon::AddonLogScale)
    return string("log-scale = ", getlogscale(addon), "; ")
end