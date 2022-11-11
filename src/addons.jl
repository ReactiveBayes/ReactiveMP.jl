abstract type AbstractAddon end
abstract type AbstractAddonProd end

import Base: string, show

function multiply_addons(new_dist, left_message, right_message)

    # fetch addons
    left_addons = getaddons(left_message)
    right_addons = getaddons(right_message)

    # perform sanity check on the length of the addons
    @assert length(left_addons) == length(right_addons) "Trying to perform computations with different lengths of addons."

    # compute addon product elementwise
    new_addons = ()
    for (addon_left, addon_right) in zip(left_addons, right_addons)
        new_addons = TupleTools.flatten(new_addons, multiply_addons(addon_left, addon_right, new_dist, left_message, right_message))
    end

    # return addons
    return new_addons
end

multiply_addons(::Any, ::Message{<:Any, Nothing}, ::Message{<:Any, Nothing}) = nothing
multiply_addons(::Any, message::Message{<:Any, <:Any}, ::Message{Missing, Nothing}) = getaddons(message)
multiply_addons(::Any, ::Message{Missing, Nothing}, message::Message{<:Any, <:Any}) = getaddons(message)
multiply_addons(::Nothing, ::Nothing, ::Any, ::Any, ::Any) = nothing
multiply_addons(::Nothing, addon::Any, ::Any, ::Missing, ::Any) = addon
multiply_addons(addon::Any, ::Nothing, ::Any, ::Any, ::Missing) = addon
multiply_addons(::Nothing, ::Nothing, ::Any, ::Missing, ::Any) = nothing
multiply_addons(::Nothing, ::Nothing, ::Any, ::Any, ::Missing) = nothing

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