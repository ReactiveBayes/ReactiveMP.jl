
import Base: length, map
import Base: @propagate_inbounds
import Rocket: tap

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
    left        :: LazyObservable{Missing}
    right       :: LazyObservable{Missing}
    cache_left  :: Message
    cache_right :: Message

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

"""
    EqualityChain
"""
struct EqualityChain{P, F}
    length     :: Int
    nodes      :: Vector{EqualityNode}
    inputmsgs  :: Vector{MessageObservable{AbstractMessage}}
    cacheleft  :: BitVector
    cacheright :: BitVector
    pipeline   :: P
    prod_fn    :: F

    function EqualityChain(inputmsgs::Vector{MessageObservable{AbstractMessage}}, pipeline::P, prod_fn::F) where {P, F}
        n = length(inputmsgs)
        nodes = map(_ -> EqualityNode(), 1:n)
        return new{P, F}(n, nodes, inputmsgs, falses(n), falses(n), pipeline, prod_fn)
    end
end

Base.length(chain::EqualityChain) = chain.length

prod(chain::EqualityChain, left, right) = chain.prod_fn((left, right))

getpipeline(chain::EqualityChain) = chain.pipeline

@propagate_inbounds getnode(chain::EqualityChain, node_index) = chain.nodes[node_index]

__check_indices(::EqualityLeftOutbound, chain::EqualityChain, node_index)  = 1 < node_index <= length(chain)
__check_indices(::EqualityRightOutbound, chain::EqualityChain, node_index) = 1 <= node_index < length(chain)

@propagate_inbounds getoutbound(type::EqualityNodeOutboundType, chain::EqualityChain, node_index) = __check_indices(type, chain, node_index) ? getoutbound(type, getnode(chain, node_index)) : (of(missing))
@propagate_inbounds getinbound(chain::EqualityChain, node_index)                                  = chain.inputmsgs[node_index]

@propagate_inbounds iscached(::EqualityLeftOutbound, chain::EqualityChain, node_index)  = chain.cacheleft[node_index]
@propagate_inbounds iscached(::EqualityRightOutbound, chain::EqualityChain, node_index) = chain.cacheright[node_index]

@propagate_inbounds setcache!(::EqualityLeftOutbound, chain::EqualityChain, node_index)  = chain.cacheleft[node_index] = true
@propagate_inbounds setcache!(::EqualityRightOutbound, chain::EqualityChain, node_index) = chain.cacheright[node_index] = true

@propagate_inbounds setcache!(::EqualityLeftOutbound, chain::EqualityChain, range::OrdinalRange)  = fill_bitarray!(view(chain.cacheleft, forward_range(range)), true)
@propagate_inbounds setcache!(::EqualityRightOutbound, chain::EqualityChain, range::OrdinalRange) = fill_bitarray!(view(chain.cacheright, forward_range(range)), true)

@propagate_inbounds function getcache(type::EqualityNodeOutboundType, chain::EqualityChain, node_index)
    if __check_indices(type, chain, node_index)
        return getcache(type, getnode(chain, node_index))
    else
        return Message(missing, true, true)
    end
end

nextindex(::EqualityLeftOutbound, node_index)  = node_index + 1
nextindex(::EqualityRightOutbound, node_index) = node_index - 1

@propagate_inbounds first_unmaterialized_index(::EqualityLeftOutbound, chain::EqualityChain, node_index)::Int  = something(findfirst(view(chain.cacheleft, node_index:length(chain))), length(chain) - (node_index - 1)) + (node_index - 1)
@propagate_inbounds first_unmaterialized_index(::EqualityRightOutbound, chain::EqualityChain, node_index)::Int = something(findlast(view(chain.cacheright, 1:node_index)), 1)

@propagate_inbounds precompute_range(type::EqualityLeftOutbound, chain::EqualityChain, node_index)  = first_unmaterialized_index(type, chain, node_index):-1:node_index
@propagate_inbounds precompute_range(type::EqualityRightOutbound, chain::EqualityChain, node_index) = first_unmaterialized_index(type, chain, node_index):node_index

@propagate_inbounds function materialize!(type::EqualityNodeOutboundType, chain::EqualityChain, node_index)
    if __check_indices(type, chain, node_index)
        node = getnode(chain, node_index)
        if iscached(type, chain, node_index)
            return getcache(type, node)
        else
            # precompute messages in linear fashion 
            range = precompute_range(type, chain, node_index)
            for index in range
                arg1 = as_message(getrecent(getinbound(chain, index)))
                arg2 = as_message(getcache(type, chain, nextindex(type, index)))
                result = prod(chain, arg1, arg2)
                setcache!(type, getnode(chain, index), result)
            end
            setcache!(type, chain, range)
            return materialize!(type, chain, node_index)
        end
    else
        return Message(missing, true, true)
    end
end

##

struct ChainInvalidationCallback
    index      :: Int
    cacheleft  :: BitVector
    cacheright :: BitVector

    function ChainInvalidationCallback(index::Int, chain::EqualityChain)
        return new(index, chain.cacheleft, chain.cacheright)
    end
end

Rocket.tap(callback::ChainInvalidationCallback) = Rocket.TapOperator{ChainInvalidationCallback}(callback)

function (callback::ChainInvalidationCallback)(_)
    fill_bitarray!(view(callback.cacheleft, firstindex(callback.cacheleft):(callback.index)), false)
    fill_bitarray!(view(callback.cacheright, (callback.index):lastindex(callback.cacheright)), false)
end

## 

struct ChainOutboundMapping
    index::Int
    chain::EqualityChain
end

function (mapping::ChainOutboundMapping)(_)
    from_left  = materialize!(EqualityRightOutbound(), mapping.chain, nextindex(EqualityRightOutbound(), mapping.index))
    from_right = materialize!(EqualityLeftOutbound(), mapping.chain, nextindex(EqualityLeftOutbound(), mapping.index))
    return as_message(prod(mapping.chain, from_left, from_right))
end

Base.map(::Type{Message}, mapping::ChainOutboundMapping) = Rocket.MapOperator{Message, ChainOutboundMapping}(mapping)

function initialize!(chain::EqualityChain, outputmsgs::AbstractVector)
    n = length(chain)

    pipeline = getpipeline(chain)

    Left  = EqualityLeftOutbound()
    Right = EqualityRightOutbound()

    @inbounds for index in 1:n
        node = getnode(chain, index)

        # As soon as we receive new inbound message - we invalidate cache for part of the chain: see ChainInvalidationCallback
        input = getinbound(chain, index) |> tap(ChainInvalidationCallback(index, chain)) |> share_recent()

        left  = combineLatestUpdates((getoutbound(Left, chain, nextindex(Left, index)), input), PushNew()) |> pipeline |> map_to(missing) |> share_recent()
        right = combineLatestUpdates((getoutbound(Right, chain, nextindex(Right, index)), input), PushNew()) |> pipeline |> map_to(missing) |> share_recent()

        setoutbound!(Left, node, left)
        setoutbound!(Right, node, right)

        from_left  = getoutbound(Right, chain, nextindex(Right, index)) # Inbound message comming from left direction  (is a right from `index - 1`)
        from_right = getoutbound(Left, chain, nextindex(Left, index))  # Inbound message comming from right direction (is a left from `index + 1`)

        connect!(outputmsgs[index], combineLatestUpdates((from_left, from_right), PushNew()) |> map(Message, ChainOutboundMapping(index, chain)))
    end

    return nothing
end
