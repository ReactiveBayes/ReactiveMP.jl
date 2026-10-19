import Base: @propagate_inbounds
import Rocket: tap

## Equality node is a special case and has a special implementation
## It should not be used during model creation but instead is a part of variable node implementation

abstract type EqualityNodeOutboundType end

struct EqualityLeftOutbound <: EqualityNodeOutboundType end
struct EqualityRightOutbound <: EqualityNodeOutboundType end

"""
    ReactiveMP.EqualityNode()

One link of an [`ReactiveMP.EqualityChain`](@ref), for one connection of a random variable: the
partial products of the inbound messages on either side of it, kept so that they are reused.

```
 <- left    -------     right ->
     ----- |   =   | ------
            -------
               |
               | outbound
```

# Fields

- `left`: the stream that says the product of this and every later inbound message can be
  computed; it emits `missing`, a notification only;
- `right`: the stream that says the product of this and every earlier inbound message can be
  computed; it emits `missing` too;
- `cache_left`: the cached product towards the left; the chain tracks whether it is valid;
- `cache_right`: the cached product towards the right; the chain tracks whether it is valid.
"""
mutable struct EqualityNode
    left::LazyObservable{Missing}
    right::LazyObservable{Missing}
    cache_left::Message
    cache_right::Message

    EqualityNode() = new(
        lazy(Missing),
        lazy(Missing),
        Message(missing, true, true),
        Message(missing, true, true),
    )
end

getoutbound(::EqualityLeftOutbound, node::EqualityNode) = node.left
getoutbound(::EqualityRightOutbound, node::EqualityNode) = node.right

setoutbound!(::EqualityLeftOutbound, node::EqualityNode, left) = set!(node.left, left)
setoutbound!(::EqualityRightOutbound, node::EqualityNode, right) = set!(node.right, right)

getcache(::EqualityLeftOutbound, node::EqualityNode) = node.cache_left
getcache(::EqualityRightOutbound, node::EqualityNode) = node.cache_right

setcache!(::EqualityLeftOutbound, node::EqualityNode, cache::Message) = node.cache_left = cache
setcache!(::EqualityRightOutbound, node::EqualityNode, cache::Message) = node.cache_right = cache

# Equality chain

"""
    ReactiveMP.EqualityChain(inputmsgs::Vector{MessageObservable{AbstractMessage}}, postprocessor, variable::AbstractVariable, context::MessageProductContext)

How a random variable with more than one connection computes its outbound messages. The message
to connection `i` is the product of the inbound messages on every other connection, which the
chain computes from two partial products, of the messages before `i` and of those after it. The
partial products are cached in the [`ReactiveMP.EqualityNode`](@ref)s and shared by the outbound
messages, which matters for a variable of high degree; a new inbound message on connection `i`
invalidates only the partial products that contain it.

A partial product is a part of a message, not a message: it is multiplied by `context`'s fold
strategy alone, pairwise with [`ReactiveMP.compute_product_of_two_messages`](@ref), which applies
the form constraint under [`FormConstraintCheckEach`](@ref). The product of the two parts is the
message, and is computed with [`ReactiveMP.compute_product_of_messages`](@ref): under
[`FormConstraintCheckLast`](@ref) the form constraint applies to it once, and the callbacks see
one whole product per outbound message.

# Arguments

- `inputmsgs`: the variable's inbound message streams, one per connection;
- `postprocessor`: the stream postprocessor applied to the chain's streams, or `nothing`;
- `variable`: the variable the products are for, which the callbacks receive;
- `context`: how the messages are multiplied; a random variable gives its
  `prod_context_for_message_computation`.
"""
struct EqualityChain{P, V, C}
    length::Int
    nodes::Vector{EqualityNode}
    inputmsgs::Vector{MessageObservable{AbstractMessage}}
    cacheleft::Vector{Bool}
    cacheright::Vector{Bool}
    postprocessor::P
    variable::V
    context::C

    function EqualityChain(
            inputmsgs::Vector{MessageObservable{AbstractMessage}},
            postprocessor::P,
            variable::V,
            context::C,
        ) where {P, V <: AbstractVariable, C <: MessageProductContext}
        n = length(inputmsgs)
        nodes = map(_ -> EqualityNode(), 1:n)
        return new{P, V, C}(
            n, nodes, inputmsgs, fill(false, n), fill(false, n), postprocessor, variable, context
        )
    end
end

Base.length(chain::EqualityChain) = chain.length

partial_product(chain::EqualityChain, left, right) =
    as_message(compute_product_of_messages(chain.context.fold_strategy, chain.variable, chain.context, (left, right)))
outbound_product(chain::EqualityChain, left, right) =
    compute_product_of_messages(chain.variable, chain.context, (left, right))

getpostprocessor(chain::EqualityChain) = chain.postprocessor

@propagate_inbounds getnode(chain::EqualityChain, node_index) =
    chain.nodes[node_index]

__check_indices(::EqualityLeftOutbound, chain::EqualityChain, node_index) = 1 < node_index <= length(chain)
__check_indices(::EqualityRightOutbound, chain::EqualityChain, node_index) = 1 <= node_index < length(chain)

@propagate_inbounds getoutbound(type::EqualityNodeOutboundType, chain::EqualityChain, node_index) = __check_indices(type, chain, node_index) ? getoutbound(type, getnode(chain, node_index)) : (of(missing))
@propagate_inbounds getinbound(chain::EqualityChain, node_index) = chain.inputmsgs[node_index]

@propagate_inbounds iscached(::EqualityLeftOutbound, chain::EqualityChain, node_index) = chain.cacheleft[node_index]
@propagate_inbounds iscached(::EqualityRightOutbound, chain::EqualityChain, node_index) = chain.cacheright[node_index]

@propagate_inbounds setcache!(::EqualityLeftOutbound, chain::EqualityChain, node_index) = chain.cacheleft[node_index] = true
@propagate_inbounds setcache!(::EqualityRightOutbound, chain::EqualityChain, node_index) = chain.cacheright[node_index] = true

@propagate_inbounds setcache!(::EqualityLeftOutbound, chain::EqualityChain, range::OrdinalRange) = fill!(view(chain.cacheleft, forward_range(range)), true)
@propagate_inbounds setcache!(::EqualityRightOutbound, chain::EqualityChain, range::OrdinalRange) = fill!(view(chain.cacheright, forward_range(range)), true)

@propagate_inbounds function getcache(
        type::EqualityNodeOutboundType, chain::EqualityChain, node_index
    )
    if __check_indices(type, chain, node_index)
        return getcache(type, getnode(chain, node_index))
    else
        return Message(missing, true, true)
    end
end

nextindex(::EqualityLeftOutbound, node_index) = node_index + 1
nextindex(::EqualityRightOutbound, node_index) = node_index - 1

# The node the partial product starts from: the one next to the nearest cached product, which is
# still valid, since a new inbound message invalidates every product that contains it; with
# nothing cached, the end of the chain.
@propagate_inbounds first_unmaterialized_index(::EqualityLeftOutbound, chain::EqualityChain, node_index)::Int = something(findfirst(view(chain.cacheleft, node_index:length(chain))), length(chain) - node_index + 2) + node_index - 2
@propagate_inbounds first_unmaterialized_index(::EqualityRightOutbound, chain::EqualityChain, node_index)::Int = something(findlast(view(chain.cacheright, 1:node_index)), 0) + 1

@propagate_inbounds precompute_range(type::EqualityLeftOutbound, chain::EqualityChain, node_index) = first_unmaterialized_index(type, chain, node_index):-1:node_index
@propagate_inbounds precompute_range(type::EqualityRightOutbound, chain::EqualityChain, node_index) = first_unmaterialized_index(type, chain, node_index):node_index

@propagate_inbounds function materialize!(
        type::EqualityNodeOutboundType, chain::EqualityChain, node_index
    )
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
                result = partial_product(chain, arg1, arg2)
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
    index::Int
    cacheleft::Vector{Bool}
    cacheright::Vector{Bool}

    function ChainInvalidationCallback(index::Int, chain::EqualityChain)
        return new(index, chain.cacheleft, chain.cacheright)
    end
end

Rocket.tap(callback::ChainInvalidationCallback) =
    Rocket.TapOperator{ChainInvalidationCallback}(callback)

function (callback::ChainInvalidationCallback)(_)
    fill!(
        view(
            callback.cacheleft, firstindex(callback.cacheleft):(callback.index)
        ),
        false,
    )
    return fill!(
        view(
            callback.cacheright, (callback.index):lastindex(callback.cacheright)
        ),
        false,
    )
end

##

struct ChainOutboundMapping{C <: EqualityChain}
    index::Int
    chain::C
end

function (mapping::ChainOutboundMapping)(_)
    from_left = materialize!(EqualityRightOutbound(), mapping.chain, nextindex(EqualityRightOutbound(), mapping.index))
    from_right = materialize!(EqualityLeftOutbound(), mapping.chain, nextindex(EqualityLeftOutbound(), mapping.index))
    return outbound_product(mapping.chain, from_left, from_right)
end

Base.map(::Type{Message}, mapping::M) where {M <: ChainOutboundMapping} =
    Rocket.MapOperator{Message, M}(mapping)

function initialize!(chain::EqualityChain, outputmsgs::AbstractVector)
    n = length(chain)

    postprocessor = getpostprocessor(chain)

    Left = EqualityLeftOutbound()
    Right = EqualityRightOutbound()

    @inbounds for index in 1:n
        node = getnode(chain, index)

        # As soon as we receive new inbound message - we invalidate cache for part of the chain: see ChainInvalidationCallback
        input =
            getinbound(chain, index) |>
            tap(ChainInvalidationCallback(index, chain)) |>
            share_recent()

        left = combineLatestUpdates(
            (getoutbound(Left, chain, nextindex(Left, index)), input), PushNew()
        )
        left = postprocess_stream_of_outbound_messages(postprocessor, left)
        left = left |> map_to(missing) |> share_recent()

        right = combineLatestUpdates(
            (getoutbound(Right, chain, nextindex(Right, index)), input),
            PushNew(),
        )
        right = postprocess_stream_of_outbound_messages(postprocessor, right)
        right = right |> map_to(missing) |> share_recent()

        setoutbound!(Left, node, left)
        setoutbound!(Right, node, right)

        from_left = getoutbound(Right, chain, nextindex(Right, index)) # Inbound message comming from left direction  (is a right from `index - 1`)
        from_right = getoutbound(Left, chain, nextindex(Left, index))  # Inbound message comming from right direction (is a left from `index + 1`)

        outputmsg = combineLatestUpdates(
            (from_left, from_right),
            PushNew(),
            Message,
            ChainOutboundMapping(index, chain),
            reset_vstatus,
        )

        connect!(outputmsgs[index], outputmsg)
    end

    return nothing
end
