export AddonLogScale

import Base: prod, string

abstract type AbstractAddon end
abstract type AbstractAddonProd end

struct AddonLogScale{T} <: AbstractAddon
    logscale :: T
end
AddonLogScale() = AddonLogScale(nothing)
struct AddonProdLogScale <: AbstractAddonProd end

getlogscale(message::Message) = getlogscale(getaddons(message))
getlogscale(addon::AddonLogScale) = addon.logscale
getlogscale(::Any) = 0
getlogscale(addons::Tuple{<:AbstractAddon}) = mapreduce(getlogscale, +, addons)

function prod(left_addon::AddonLogScale, right_addon::AddonLogScale, new_dist::Distribution, left_dist::Distribution, right_dist::Distribution)
    left_logscale = getlogscale(left_addon)
    right_logscale = getlogscale(right_addon)
    new_logscale = prod(AddonProdLogScale(), new_dist, left_dist, right_dist)
    return AddonLogScale(left_logscale + right_logscale + new_logscale)
end

function string(addons::NTuple{N, <:AbstractAddon}) where { N }
    if length(addons) == 0
        return "no addons"
    end
    str = ""
    for addon in addons
        str = string(str, string(addon))
    end
    return str
end

function string(addon::AddonLogScale)
    return string("log-scale = ", getlogscale(addon), "; ")
end