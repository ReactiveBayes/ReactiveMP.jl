
import Base: length

## Equality node is a special case and has a special implementation
## It should not be used during model creation but instead is a part of variable node implementation

## Equality node

mutable struct EqualityNode 
    index        :: Int
    cache_left   :: Union{Nothing, Message}
    cache_right  :: Union{Nothing, Message}
    left         :: LazyObservable{Int}
    right        :: LazyObservable{Int}
    inbound      :: MessageObservable{AbstractMessage}
end

EqualityNode(index::Int, inbound::MessageObservable{AbstractMessage}) = EqualityNode(index, nothing, nothing, lazy(Int), lazy(Int), inbound)

getindex(node::EqualityNode)   = node.index
getleft(node::EqualityNode)    = node.left
getright(node::EqualityNode)   = node.right
getinbound(node::EqualityNode) = node.inbound

setleft!(node::EqualityNode, left)   = set!(node.left, left)
setright!(node::EqualityNode, right) = set!(node.right, right)

get_cache_left(node::EqualityNode)  = node.cache_left
get_cache_right(node::EqualityNode) = node.cache_right

set_cache_left!(node::EqualityNode, cache)  = node.cache_left = cache
set_cache_right!(node::EqualityNode, cache) = node.cache_right = cache

function invalidate!(node::EqualityNode)
    node.cache_left  = nothing
    node.cache_right = nothing
    return nothing
end

# Equality chain

struct EqualityChain{F}
    length  :: Int
    nodes   :: Vector{EqualityNode}
    prod_fn :: F
end

EqualityChain(length::Int, inbounds::Vector{MessageObservable{AbstractMessage}}, prod_fn::F) where { F } = EqualityChain(length, map(i -> EqualityNode(i[1], i[2]), enumerate(inbounds)), prod_fn)

Base.length(chain::EqualityChain) = chain.length

prod(chain::EqualityChain, left, right) = chain.prod_fn((left, right))

invalidate!(chain::EqualityChain) = foreach(invalidate!, chain.nodes)

getleft(chain::EqualityChain, node_index::Int)  = (1 < node_index <= length(chain)) ? (@inbounds getleft(chain.nodes[node_index])) : (of(node_index))
getright(chain::EqualityChain, node_index::Int) = (1 <= node_index < length(chain)) ? (@inbounds getright(chain.nodes[node_index])) : (of(node_index))

function materialize_left!(chain::EqualityChain, node_index::Int)
    if (1 < node_index <= length(chain))
        node   = @inbounds chain.nodes[node_index]
        cached = get_cache_left(node)
        if cached !== nothing 
            return cached
        end
        # Compute cache and save
        result = prod(chain, as_message(getrecent(getinbound(node))), materialize_left!(chain, node_index + 1))
        set_cache_left!(node, result)
        return result
    else 
        return Message(missing, true, true)
    end
end

function materialize_right!(chain::EqualityChain, node_index::Int)
    if (1 <= node_index < length(chain))
        node   = @inbounds chain.nodes[node_index]
        cached = get_cache_right(node)
        if cached !== nothing 
            return cached
        end
        # Compute cache and save
        result = prod(chain, as_message(getrecent(getinbound(node))), materialize_right!(chain, node_index - 1))
        set_cache_right!(node, result)
        return result
    else 
        return Message(missing, true, true)
    end
end

function activate!(model, chain::EqualityChain, inputmsgs::AbstractVector, outputmsgs::AbstractVector)
    n = length(chain)

    @inbounds for index in 1:n
        from_left  = getright(chain, index - 1) # Inbound message comming from left direction  (is a right from `index - 1`)
        from_right = getleft(chain, index + 1)  # Inbound message comming from right direction (is a left from `index + 1`)

        outbound_mapping = let chain = chain 
            (indices) -> as_message(prod(chain, materialize_right!(chain, indices[1]), materialize_left!(chain, indices[2])))
        end

        connect!(outputmsgs[index], combineLatest((from_left, from_right), PushNew()) |> map(Message, outbound_mapping))

        node = chain.nodes[index]
        node_inbound = inputmsgs[index] |> tap(_ -> invalidate!(chain)) |> share_recent()

        node_left  = combineLatest((getleft(chain, index + 1), node_inbound), PushNew()) |> schedule_on(global_reactive_scheduler(getoptions(model)))
        node_right = combineLatest((getright(chain, index - 1), node_inbound), PushNew()) |> schedule_on(global_reactive_scheduler(getoptions(model)))

        setleft!(node, node_left |> map_to(index) |> share_recent())
        setright!(node, node_right |> map_to(index) |> share_recent())
    end

    return nothing
end