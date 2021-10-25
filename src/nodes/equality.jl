
import Base: length
import Base: @propagate_inbounds

## Equality node is a special case and has a special implementation
## It should not be used during model creation but instead is a part of variable node implementation

abstract type EqualityNodeOutboundType end

struct EqualityLeftOutbound <: EqualityNodeOutboundType end
struct EqualityRightOutbound <: EqualityNodeOutboundType end

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

@propagate_inbounds getnode(chain::EqualityChain, node_index) = chain.nodes[node_index]

function invalidate!(chain::EqualityChain, index) 
    fill_bitarray!(view(chain.cacheleft, firstindex(chain.cacheleft):index), false)
    fill_bitarray!(view(chain.cacheright, index:lastindex(chain.cacheright)), false)
    return nothing
end

__check_indices(::EqualityLeftOutbound, chain::EqualityChain, node_index)  = 1 < node_index <= length(chain)
__check_indices(::EqualityRightOutbound, chain::EqualityChain, node_index) = 1 <= node_index < length(chain)

@propagate_inbounds getoutbound(type::EqualityNodeOutboundType, chain::EqualityChain, node_index) = __check_indices(type, chain, node_index) ? getoutbound(type, getnode(chain, node_index)) : (of(missing))
@propagate_inbounds getinbound(chain::EqualityChain, node_index)                                  = chain.inputmsgs[node_index]

@propagate_inbounds iscached(::EqualityLeftOutbound, chain::EqualityChain, node_index)  = chain.cacheleft[node_index]
@propagate_inbounds iscached(::EqualityRightOutbound, chain::EqualityChain, node_index) = chain.cacheright[node_index]

@propagate_inbounds setcache!(::EqualityLeftOutbound, chain::EqualityChain, node_index)  = chain.cacheleft[node_index] = true
@propagate_inbounds setcache!(::EqualityRightOutbound, chain::EqualityChain, node_index) = chain.cacheright[node_index] = true

nextindex(::EqualityLeftOutbound, node_index)  = node_index + 1
nextindex(::EqualityRightOutbound, node_index) = node_index - 1

@propagate_inbounds function materialize!(type::EqualityNodeOutboundType, chain::EqualityChain, node_index)
    if __check_indices(type, chain, node_index)
        node = getnode(chain, node_index)
        if iscached(type, chain, node_index)
            return getcache(type, node)
        end
        arg1 = as_message(getrecent(getinbound(chain, node_index)))
        arg2 = as_message(materialize!(type, chain, nextindex(type, node_index)))
        result = prod(chain, arg1, arg2)
        setcache!(type, node, result)
        setcache!(type, chain, node_index)
        return result
    else
        return Message(missing, true, true)
    end
end

function activate!(model, chain::EqualityChain, outputmsgs::AbstractVector)
    n = length(chain)

    pipeline = schedule_on(global_reactive_scheduler(getoptions(model)))

    Left  = EqualityLeftOutbound()
    Right = EqualityRightOutbound() 

    @inbounds for index in 1:n
        node = chain.nodes[index]

        input = chain.inputmsgs[index] |> tap((_) -> invalidate!(chain, index)) |> share_recent()
        left  = combineLatestUpdates((getoutbound(Left, chain, nextindex(Left, index)), input), PushNew()) |> pipeline |> map_to(missing) |> share_recent()
        right = combineLatestUpdates((getoutbound(Right, chain, nextindex(Right, index)), input), PushNew()) |> pipeline |> map_to(missing) |> share_recent()

        setoutbound!(Left, node, left)
        setoutbound!(Right, node, right)

        from_left  = getoutbound(Right, chain, nextindex(Right, index)) # Inbound message comming from left direction  (is a right from `index - 1`)
        from_right = getoutbound(Left, chain, nextindex(Left, index))  # Inbound message comming from right direction (is a left from `index + 1`)

        outbound_mapping = let chain = chain, index = index
            (_) -> as_message(prod(chain, materialize!(Right, chain, index - 1), materialize!(Left, chain, index + 1)))
        end

        connect!(outputmsgs[index], combineLatestUpdates((from_left, from_right), PushNew()) |> map(Message, outbound_mapping))
    end

    return nothing
end