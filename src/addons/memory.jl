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

struct AddonMemoryMessage{T, I, F}
    messages   :: T
    interfaces :: I
    factornode :: F
end

struct AddonMemoryProd{T}
    messages :: Vector{T}
end


function multiply_addons(::AddonMemory{<:AddonMemoryMessage}, ::AddonMemory, ::Distribution, left_message::Message, right_message::Message)

    # construct new memory object
    memory = AddonMemoryProd(Message[left_message, right_message])

    # return addon
    return AddonMemory(memory)

end

function multiply_addons(::AddonMemory{<:AddonMemoryProd}, ::AddonMemory, ::Distribution, left_message::Message, right_message::Message)

    # add new message to existing memory object
    push!(getmemory(left_message).messages, right_message)
    
    # return addon
    return getmemoryaddon(left_message)

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

function show(io::IO, addon::AddonMemoryProd)    
    indent = get(io, :indent, 0)
    println(io, ' '^indent, "Product memory:")
    for message in addon.messages
        show(IOContext(io, :indent => indent+4), message)
    end
end