import MacroTools: @capture
import Base: string, show, +
import TupleTools

abstract type AbstractAddon end
abstract type AbstractAddonProd end

multiply_addons(::Nothing, ::Nothing, ::Any, ::Any, ::Any) = nothing
multiply_addons(::Nothing, addon::Any, ::Any, ::Missing, ::Any) = addon
multiply_addons(addon::Any, ::Nothing, ::Any, ::Any, ::Missing) = addon
multiply_addons(::Nothing, ::Nothing, ::Any, ::Missing, ::Any) = nothing
multiply_addons(::Nothing, ::Nothing, ::Any, ::Any, ::Missing) = nothing
multiply_addons(::Nothing, ::Nothing, ::Any, ::Missing, ::Missing) = nothing

function multiply_addons(left_addons::Tuple, right_addons::Tuple, new_dist, left_dist, right_dist)

    # perform sanity check on the length of the addons
    @assert length(left_addons) == length(right_addons) "Trying to perform computations with different lengths of addons."

    # compute addon product elementwise
    return map(left_addons, right_addons) do left_addon, right_addon
        multiply_addons(left_addon, right_addon, new_dist, left_dist, right_dist)
    end
end

# Nice functionality, allows to write `addons = Addon1() + Addon2() + ...`
+(left::AbstractAddon, right::AbstractAddon) = (left, right)
+(left::NTuple{N, AbstractAddon}, right::AbstractAddon) where {N} = (left..., right)
+(left::AbstractAddon, right::NTuple{N, AbstractAddon}) where {N} = (left, right...)

macro invokeaddon(Type, callback)
    # invoke addon macro can be executed only inside the @rule macro 
    index = gensym(:index)
    addon = gensym(:addon)
    value = gensym(:value)
    body = quote
        # First we check is the `_addons` field of the `@rule` macro has the associated addon enabled
        # To do that we check if the type of the addon is present in the `_addons` tuple
        _addons = if !isnothing(_addons)
            # If the specified addons is enabled we find its index and replace the value with the specified body
            local $index = findnext((addon) -> addon isa $(Type), _addons, 1)
            if !isnothing($index)
                local $addon = () -> $(Type)($(callback))
                local $value = $(addon)()
                # Here we replace the previous value of the addon at the specified index
                ReactiveMP.TupleTools.insertat(_addons, $index, ($value,))
            else
                _addons
            end
        else
            # If the addon is not present we simply return the result
            _addons
        end
    end
    return esc(body)
end

# TODO: This method is invoked on empty tuples, which messes up printing of tuples
# function string(addons::NTuple{N, AbstractAddon}) where {N}
#     if length(addons) == 0
#         return "no addons"
#     end
#     str = ""
#     for addon in addons
#         str = string(str, string(addon))
#     end
#     return str
# end

# show(io::IO, addons::NTuple{N, AbstractAddon}) where {N} = print(io, string(addons))
