export AddonDebug

"""
    AddonDebug(f :: Function)

This addon calls the function `f` over the output of the message mapping and products. The result is expected to be boolean and when returning true, it will throw an error with the debug information. Common applications of this addon are to check for NaNs and Infs in the messages and marginals. 

## Example
```julia
checkfornans(x) = isnan(x)
checkfornans(x::AbstractArray) = any(checkfornans.(x))
checkfornans(x::Tuple) = any(checkfornans.(x))

addons = (AddonDebug(dist -> checkfornans(params(dist))),)
```
"""
struct AddonDebug <: AbstractAddon
    f::Function
end

AddonDebug() = AddonDebug(nothing)

getdebugaddon(addons::NTuple{N, AbstractAddon}) where {N} = first(filter(x -> typeof(x) <: AddonDebug, addons))

(addon::AddonDebug)(x) = addon.f(x)

function message_mapping_addon(addon::AddonDebug, mapping, messages, marginals, result)
    if addon(result)

        #  create error message
        msg = "Debug addon triggered:\n"
        msg *= "Mapping:\n"
        msg *= "At the node: " * string(message_mapping_fform(mapping)) * "\n"
        msg *= "Towards interface: " * string(mapping.vtag) * "\n"
        msg *= "With local constraint: " * string(mapping.vconstraint) * "\n"
        if !isnothing(mapping.meta)
            msg *= "With meta: " * string(mapping.meta) * "\n"
        end
        if !isnothing(mapping.addons)
            msg *= "With addons: " * string(mapping.addons) * "\n"
        end
        if !isnothing(messages)
            msg *= "Incoming messages:\n"
            for message in messages
                msg *= string(message) * "\n"
            end
        end
        if !isnothing(marginals)
            msg *= "Incoming marginals:\n"
            for marginal in marginals
                msg *= string(marginal) * "\n"
            end
        end
        msg *= "Result:\n"
        msg *= string(result)

        #  throw error
        return error(msg)
    end
    return addon
end

function multiply_addons(left_addon::AddonDebug, right_addon::AddonDebug, new_dist, left_dist, right_dist)
    if left_addon(new_dist)

        # create error message
        msg = "Debug addon triggered:\n"
        msg *= "Incoming distributions: \n"
        msg *= string(left_dist) * "\n"
        msg *= string(right_dist) * "\n"
        msg *= "Resulting distribution: \n"
        msg *= string(new_dist)

        # throw error
        return error(msg)
    end
    return left_addon
end
