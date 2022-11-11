export AddonMemory, getmemory

import Base: prod, string, show

struct AddonMemory{T} <: AbstractAddon
    memory::T
end
AddonMemory() = AddonMemory(nothing)
struct AddonProdMemory <: AbstractAddonProd end

getmemory(addon::AddonMemory) = addon.memory
getmemory(addons::NTuple{N, AbstractAddon}) where {N} = first(filter(x -> typeof(x) <: AddonMemory, addons)).memory

struct AddonMemoryMessage{T, I, F}
    messages   :: T
    interfaces :: I
    factornode :: F
end

struct AddonMemoryMarginal{T}
    messages :: Vector{T}
end


function multiply_addons(::AddonMemory, ::AddonMemory, ::Distribution, left_message::Message, right_message::Message)

    # construct new memory object
    memory = AddonMemoryMarginal([left_message, right_message])

    # return addon
    return AddonMemory(memory)

end

function string(::AddonMemory)
    return string("memory present; ")
end

show(io::IO, addon::AddonMemory) = show(io, addon.memory)

function show(io::IO, addon::AddonMemoryMessage)
    indent = get(io, :indent, 0)
    println(io, ' '^indent, "Message memory:")
    for message in addon.messages
        show(IOContext(io, :indent => indent+4), message)
    end
    println(io, ' '^indent, "At the node: ", addon.factornode)
    println(io, ' '^indent, "At interfaces: ", addon.interfaces)
end

function show(io::IO, addon::AddonMemoryMarginal)    
    indent = get(io, :indent, 0)
    println(io, ' '^indent, "Marginal memory:")
    for message in addon.messages
        show(IOContext(io, :indent => indent+4), message)
    end
end