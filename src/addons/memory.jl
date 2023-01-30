export AddonMemory, getmemory

import Base: prod, string, show

struct AddonMemory{T} <: AbstractAddon
    memory::T
end

AddonMemory() = AddonMemory(nothing)

struct AddonProdMemory <: AbstractAddonProd end

getmemoryaddon(addons::NTuple{N, AbstractAddon}) where {N} = first(filter(x -> typeof(x) <: AddonMemory, addons))
getmemory(addon::AddonMemory) = addon.memory
getmemory(addons::NTuple{N, AbstractAddon}) where {N} = getmemoryaddon(addons).memory

struct AddonMemoryMessageMapping{M <: MessageMapping, S, L, R}
    mapping   :: M
    messages  :: S
    marginals :: L
    result    :: R
end

struct AddonMemoryProd{T}
    mappings::Vector{T}
end

function message_mapping_addon(::AddonMemory{Nothing}, mapping, messages, marginals, result)
    return AddonMemory(AddonMemoryMessageMapping(mapping, messages, marginals, result))
end

function multiply_addons(left_addon::AddonMemory, right_addon::AddonMemory, new_dist, left_dist, right_dist)
    return AddonMemory(prod(AddonProdMemory(), getmemory(left_addon), getmemory(right_addon)))
end

function prod(::AddonProdMemory, left::AddonMemoryMessageMapping, right::AddonMemoryMessageMapping)
    return AddonMemoryProd(Any[left, right])
end

function prod(::AddonProdMemory, left::AddonMemoryMessageMapping, right::AddonMemoryProd)
    pushfirst!(right.mappings, left)
    return right
end

function prod(::AddonProdMemory, left::AddonMemoryProd, right::AddonMemoryMessageMapping)
    push!(left.mappings, right)
    return left
end

function prod(::AddonProdMemory, left::AddonMemoryProd, right::AddonMemoryProd)
    append!(left.mappings, right.mappings)
    return left
end

function string(::AddonMemory)
    return string("memory present; ")
end

show(io::IO, addon::AddonMemory) = print(io, string("AddonMemory(", addon.memory, ")"))

function show(io::IO, addon::AddonMemoryMessageMapping)
    indent = get(io, :indent, 0)
    println(io, ' ', "Message mapping memory:")
    println(io, ' '^indent, "At the node: ", functionalform(addon.mapping.factornode))
    println(io, ' '^indent, "Towards interface: ", addon.mapping.vtag)
    println(io, ' '^indent, "With local constraint: ", addon.mapping.vconstraint)
    if !isnothing(addon.mapping.meta)
        println(io, ' '^indent, "With meta: ", addon.mapping.meta)
    end
    if !isnothing(addon.mapping.addons)
        println(io, ' '^indent, "With addons: ", addon.mapping.addons)
    end
    if !isnothing(addon.messages)
        println(io, ' '^indent, "With input messages on ", addon.mapping.msgs_names, " edges: ", addon.messages)
    end
    if !isnothing(addon.marginals)
        println(io, ' '^indent, "With input marginals on ", addon.mapping.marginals_names, " edges: ", addon.marginals)
    end
    println(io, ' '^indent, "With the result: ", addon.result)
end

function show(io::IO, addon::AddonMemoryProd)
    indent = get(io, :indent, 0)
    println(io, ' '^indent, "Product memory:")
    for message in addon.mappings
        show(IOContext(io, :indent => indent + 4), message)
    end
end
