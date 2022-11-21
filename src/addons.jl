abstract type AbstractAddon end
abstract type AbstractAddonProd end

import Base: string, show

multiply_addons(::Nothing, ::Nothing, ::Any, ::Any, ::Any) = nothing
multiply_addons(::Nothing, addon::Any, ::Any, ::Missing, ::Any) = addon
multiply_addons(addon::Any, ::Nothing, ::Any, ::Any, ::Missing) = addon
multiply_addons(::Nothing, ::Nothing, ::Any, ::Missing, ::Any) = nothing
multiply_addons(::Nothing, ::Nothing, ::Any, ::Any, ::Missing) = nothing

function multiply_addons(left_addons::Tuple, right_addons::Tuple, new_dist, left_dist, right_dist)

    # perform sanity check on the length of the addons
    @assert length(left_addons) == length(right_addons) "Trying to perform computations with different lengths of addons."

    # compute addon product elementwise
    return map(left_addons, right_addons) do left_addon, right_addon
        multiply_addons(left_addon, right_addon, new_dist, left_dist, right_dist)
    end
end

function string(addons::NTuple{N, AbstractAddon}) where {N}
    if length(addons) == 0
        return "no addons"
    end
    str = ""
    for addon in addons
        str = string(str, string(addon))
    end
    return str
end

show(io::IO, addons::NTuple{N, AbstractAddon}) where {N} = print(io, string(addons))
