
import Base: length

## Equality node is a special case and has a special implementation
## It should not be used during model creation but instead is a part of variable node implementation

struct EqualityLeftOutbound end
struct EqualityRightOutbound end

"""
    EqualityNode
 
Generic idea of an equality node is to keep track of intermediate `left` and `right` messages 
in a convenient manner so they can be reused during `prod` for random variables with large `degree`.

 <- left    -------     right ->
     ----- |   =   | ------
            -------
               |
               | outbound

# Attributes:
- `left`: An observable that indicates that `left` message can be computed, always sends `missing` since we are not interested in actual value. Can be treated as notification.
- `right`: An observable that indicates that `right` message can be computed, always sends `missing` since we are not interested in actual value. Can be treated as notification.
- `cache_left`: Keeps cached value for left outbound message, node itself does not track if cache is valid or not. Cache integrity is tracked by `EqualityChain`.
- `cache_right`: Keeps cached value for right outbound message, node itself does not track if cache is valid or not. Cache integrity is tracked by `EqualityChain`.

See also: [`EqualityChain`](@ref)
"""
mutable struct EqualityNode 
    left         :: LazyObservable{Missing}
    right        :: LazyObservable{Missing}
    cache_left   :: Message
    cache_right  :: Message

    EqualityNode() = new(lazy(Missing), lazy(Missing), Message(missing, true, true), Message(missing, true, true))
end

getoutbound(::EqualityLeftOutbound, node::EqualityNode)  = node.left
getoutbound(::EqualityRightOutbound, node::EqualityNode) = node.right

setoutbound!(::EqualityLeftOutbound, node::EqualityNode, left)   = set!(node.left, left)
setoutbound!(::EqualityRightOutbound, node::EqualityNode, right) = set!(node.right, right)

getcache(::EqualityLeftOutbound, node::EqualityNode)  = node.cache_left
getcache(::EqualityRightOutbound, node::EqualityNode) = node.cache_right

setcache!(::EqualityLeftOutbound, node::EqualityNode, cache::Message)  = node.cache_left = cache
setcache!(::EqualityRightOutbound, node::EqualityNode, cache::Message) = node.cache_right = cache

# Equality chain

struct EqualityChain{F}
    length     :: Int
    nodes      :: Vector{EqualityNode}
    inputmsgs  :: Vector{MessageObservable{AbstractMessage}}
    cacheleft  :: BitVector
    cacheright :: BitVector
    prod_fn    :: F

    function EqualityChain(inputmsgs::Vector{MessageObservable{AbstractMessage}}, prod_fn::F) where { F } 
        n     = length(inputmsgs)
        nodes = map(_ -> EqualityNode(), 1:n)
        return new{F}(n, nodes, inputmsgs, falses(n), falses(n), prod_fn)     
    end
end

Base.length(chain::EqualityChain) = chain.length

prod(chain::EqualityChain, left, right) = chain.prod_fn((left, right))

function invalidate!(chain::EqualityChain, index) 
    fill_bitarray!(view(chain.cacheleft, firstindex(chain.cacheleft):index), false)
    fill_bitarray!(view(chain.cacheright, index:lastindex(chain.cacheright)), false)
    return nothing
end

getleft(chain::EqualityChain, node_index::Int)    = (1 < node_index <= length(chain)) ? (@inbounds getoutbound(EqualityLeftOutbound(), chain.nodes[node_index])) : (of(missing))
getright(chain::EqualityChain, node_index::Int)   = (1 <= node_index < length(chain)) ? (@inbounds getoutbound(EqualityRightOutbound(), chain.nodes[node_index])) : (of(missing))
getinbound(chain::EqualityChain, node_index::Int) = @inbounds chain.inputmsgs[node_index]

is_left_cached(chain, node_index::Int)  = @inbounds chain.cacheleft[node_index]
is_right_cached(chain, node_index::Int) = @inbounds chain.cacheright[node_index]

function cache_left!(chain::EqualityChain, node_index::Int, node::EqualityNode, cache) 
    setcache!(EqualityLeftOutbound(), node, cache)
    @inbounds chain.cacheleft[node_index] = true
    return nothing
end

function cache_right!(chain::EqualityChain, node_index::Int, node::EqualityNode, cache) 
    setcache!(EqualityRightOutbound(), node, cache)
    @inbounds chain.cacheright[node_index] = true
    return nothing
end

function materialize_left!(chain::EqualityChain, node_index::Int)
    if (1 < node_index <= length(chain))
        node = @inbounds chain.nodes[node_index]
        if is_left_cached(chain, node_index)
            return getcache(EqualityLeftOutbound(), node)
        end
        # Compute cache and save
        result = prod(chain, as_message(getrecent(getinbound(chain, node_index))), materialize_left!(chain, node_index + 1))
        cache_left!(chain, node_index, node, result)
        return result
    else 
        return Message(missing, true, true)
    end
end

function materialize_right!(chain::EqualityChain, node_index::Int)
    if (1 <= node_index < length(chain))
        node = @inbounds chain.nodes[node_index]
        if is_right_cached(chain, node_index)
            return getcache(EqualityRightOutbound(), node)
        end
        # Compute cache and save
        result = prod(chain, as_message(getrecent(getinbound(chain, node_index))), materialize_right!(chain, node_index - 1))
        cache_right!(chain, node_index, node, result)
        return result
    else 
        return Message(missing, true, true)
    end
end

function activate!(model, chain::EqualityChain, outputmsgs::AbstractVector)
    n = length(chain)

    pipeline = schedule_on(global_reactive_scheduler(getoptions(model)))

    @inbounds for index in 1:n
        node = chain.nodes[index]

        input = chain.inputmsgs[index] |> tap((_) -> invalidate!(chain, index)) |> share_recent()
        left  = combineLatestUpdates((getleft(chain, index + 1), input), PushNew()) |> pipeline |> map_to(missing) |> share_recent()
        right = combineLatestUpdates((getright(chain, index - 1), input), PushNew()) |> pipeline |> map_to(missing) |> share_recent()

        setoutbound!(EqualityLeftOutbound(), node, left)
        setoutbound!(EqualityRightOutbound(), node, right)

        from_left  = getright(chain, index - 1) # Inbound message comming from left direction  (is a right from `index - 1`)
        from_right = getleft(chain, index + 1)  # Inbound message comming from right direction (is a left from `index + 1`)

        outbound_mapping = let chain = chain, index = index
            (_) -> as_message(prod(chain, materialize_right!(chain, index - 1), materialize_left!(chain, index + 1)))
        end

        connect!(outputmsgs[index], combineLatestUpdates((from_left, from_right), PushNew()) |> map(Message, outbound_mapping))
    end

    return nothing
end