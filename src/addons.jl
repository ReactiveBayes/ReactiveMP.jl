export AddonLogScale
export getlogscale

import Base: prod, string

abstract type AbstractAddon end
abstract type AbstractAddonProd end

struct AddonLogScale{T} <: AbstractAddon
    logscale::T
end
AddonLogScale() = AddonLogScale(nothing)
struct AddonProdLogScale <: AbstractAddonProd end

getlogscale(message::Message) = getlogscale(getaddons(message))
getlogscale(marginal::Marginal) = getlogscale(getaddons(marginal))
getlogscale(addon::AddonLogScale) = addon.logscale
getlogscale(::AbstractAddon) = 0
getlogscale(addons::Tuple{<:AbstractAddon}) = mapreduce(getlogscale, +, addons)

function multiply_addons(left_addons, right_addons, new_dist, left_dist, right_dist)
    @assert length(left_addons) == length(right_addons) "Trying to perform computations with different lengths of addons."
    new_addons = ()
    for (addon_left, addon_right) in zip(left_addons, right_addons)
        new_addons =
            TupleTools.flatten(new_addons, multiply_addons(addon_left, addon_right, new_dist, left_dist, right_dist))
    end
    return new_addons
end
multiply_addons(::Nothing, ::Nothing, ::Any, ::Any, ::Any) = nothing
multiply_addons(::Nothing, addon::Any, ::Any, ::Missing, ::Any) = addon
multiply_addons(addon::Any, ::Nothing, ::Any, ::Any, ::Missing) = addon
multiply_addons(::Nothing, ::Nothing, ::Any, ::Missing, ::Any) = nothing
multiply_addons(::Nothing, ::Nothing, ::Any, ::Any, ::Missing) = nothing

function multiply_addons(
    left_addon::AddonLogScale,
    right_addon::AddonLogScale,
    new_dist::Distribution,
    left_dist::Distribution,
    right_dist::Distribution
)
    left_logscale = getlogscale(left_addon)
    right_logscale = getlogscale(right_addon)
    new_logscale = prod(AddonProdLogScale(), new_dist, left_dist, right_dist)
    return AddonLogScale(left_logscale + right_logscale + new_logscale)
end

function string(addons::NTuple{N, <:AbstractAddon}) where {N}
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
