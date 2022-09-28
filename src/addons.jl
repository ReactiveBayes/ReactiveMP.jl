export AddonFlagScaling

import Base: prod, string

abstract type AbstractAddon end
abstract type AbstractAddonFlag end
abstract type AbstractAddonProd end

struct AddonFlagScaling <: AbstractAddonFlag end
struct AddonScaling{T} <: AbstractAddon
    scaling :: T
end
struct AddonProdScaling <: AbstractAddonProd end

getscaling(addon::AddonScaling) = addon.scaling

function prod(left_addon::AddonScaling, right_addon::AddonScaling, new_dist::Distribution, left_dist::Distribution, right_dist::Distribution)
    left_scaling = getscaling(left_addon)
    right_scaling = getscaling(right_addon)
    new_scaling = prod(AddonProdScaling(), new_dist, left_dist, right_dist)
    return AddonScaling(left_scaling + right_scaling + new_scaling)
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

function string(addon::AddonScaling)
    return string("scaling = ", getscaling(addon), "; ")
end