
import Base: length

## Equality node is a special case and has a special implementation
## It should not be used during model creation but instead is a part of variable node implementation

## Equality node

mutable struct EqualityNode 
    index   :: Int
    left    :: Union{Nothing, Message}
    right   :: Union{Nothing, Message}
    inbound :: Any
end

getindex(node::EqualityNode) = node.index

getleft(node::EqualityNode)    = node.left
getright(node::EqualityNode)   = node.right
getinbound(node::EqualityNode) = node.inbound

setleft!(node::EqualityNode, message::Message)  = node.left  = message
setright!(node::EqualityNode, message::Message) = node.right = message

function invalidate!(node::EqualityNode)
    node.left  = nothing
    node.right = nothing
    return nothing
end

# Equality chain

struct EqualityChain{F}
    length  :: Int
    nodes   :: Vector{EqualityNode}
    prod_fn :: F
end

Base.length(chain::EqualityChain) = chain.length

function prod(chain::EqualityChain, left, right) 
    @show left, right
    chain.prod_fn((left, right))
end

invalidate!(chain::EqualityChain) = foreach(invalidate!, chain.nodes)

"""
    getleft(chain::EqualityChain, node_index::Int)

Returns (and materializes) the left outbound message from an equality node in the `chain` with a given `node_index`. 
Returns missing message in case if `node_index` is not in range.

See also: [`getright`](@ref), [`getoutbound`](@ref)
"""
function getleft(chain::EqualityChain, node_index::Int)
    if 1 < node_index <= length(chain)
        # First we check if cached value exists
        node   = @inbounds chain.nodes[node_index]
        cached = getleft(node)
        if cached !== nothing
            return cached
        end
        # Otherwise recompute and save cache
        nextleft = getleft(chain, node_index + 1)
        inbound  = getrecent(getinbound(node))
        result   = prod(chain, inbound, nextleft)
        setleft!(node, result)
        return result
    else
        return Message(missing, true, false)
    end
end

"""
    getright(chain::EqualityChain, node_index::Int)

Returns (and materializes) the right outbound message from an equality node in the `chain` with a given `node_index`. 
Returns missing message in case if `node_index` is not in range.

See also: [`getleft`](@ref), [`getoutbound`](@ref)
"""
function getright(chain::EqualityChain, node_index::Int)
    if 1 <= node_index < length(chain)
        # First we check if cached value exists
        node   = @inbounds chain.nodes[node_index]
        cached = getright(node)
        if cached !== nothing
            return cached
        end
        # Otherwise recompute and save cache
        prevright = getright(chain, node_index - 1)
        inbound   = getrecent(getinbound(node))
    
        result    = prod(chain, prevright, inbound)
        setright!(node, result)
        
        return result
    else
        return Message(missing, true, false)
    end
end

"""
    getoutbound(chain::EqualityChain, node_index::Int)

Returns the outbound message from an equality node in the `chain` with a given `node_index`. 

See also: [`getleft`](@ref), [`getright`](@ref)
"""
function messageout(chain::EqualityChain, node_index::Int)
    return prod(chain, getleft(chain, node_index), getright(chain, node_index))
end



