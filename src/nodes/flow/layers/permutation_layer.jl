## TODO: make sure that index 1 is always set to some value other than 1

export PermutationLayer
export getP, getmat
export forward, forward!, backward, backward!
export jacobian, inv_jacobian
export det_jacobian, absdet_jacobian, logdet_jacobian, logabsdet_jacobian
export detinv_jacobian, absdetinv_jacobian, logdetinv_jacobian, logabsdetinv_jacobian

import Base: eltype

@doc raw"""
The permutation layer specifies an invertible mapping ``{\bf{y}} = g({\bf{x}}) = P{\bf{x}}`` where ``P`` is a permutation matrix.

`PermutationLayer(P :: PermutationMatrix)` creates the layer structure with specified permuation matrix `P`.
`PermutationLayer(dim <: Int)` creates the layer structure with dimensionality `dim`.

### Example
```julia
layer = PermutationLayer(dim)
```
"""
struct PermutationLayer{ T } <: AbstractLayer
    P :: PermutationMatrix{T}
end

function PermutationLayer(dim::T) where { T <: Int}

    # create random permutation matrix
    P = PermutationMatrix(dim)

    # return layer
    return PermutationLayer(P)

end

# get-functions for the PermutationLayer structure
getP(layer::PermutationLayer)     = layer.P
getmat(layer::PermutationLayer)   = layer.P

# custom Base function for the PermutationLayer structure
eltype(layer::PermutationLayer{T})  where { T }              = eltype(T)
eltype(::Type{PermutationLayer{T}}) where { T }              = eltype(T)

# forward pass through the permutation layer
function _forward(layer::PermutationLayer, input::Array{T,1}) where { T <: Real } 

    # fetch variables
    P = getP(layer)

    # determine result
    result = P*input

    # return result
    return result
    
end
forward(layer::PermutationLayer, input::Array{T,1}) where { T <: Real } = _forward(layer, input)
Broadcast.broadcasted(::typeof(forward), layer::PermutationLayer, input::Array{Array{T,1},1}) where { T <: Real } = broadcast(_forward, Ref(layer), input)

# inplace forward pass through the permutation layer
function forward!(output::Array{T1,1}, layer::PermutationLayer, input::Array{T2,1}) where { T1 <: Real, T2 <: Real }

    # fetch variables
    P = getP(layer)

    # determine result
    mul!(output, P, input)
    
end

# backward pass through the permutation layer
function _backward(layer::PermutationLayer, output::Array{T,1}) where { T <: Real }

    # fetch variables
    P = getP(layer)

    # determine result
    result = P'*output

    # return result
    return result
    
end
backward(layer::PermutationLayer, output::Array{T,1}) where { T <: Real } = _backward(layer, output)
Broadcast.broadcasted(::typeof(backward), layer::PermutationLayer, output::Array{Array{T,1},1}) where { T <: Real } = broadcast(_backward, Ref(layer), output)

# inplace backward pass through the additive coupling layer
function backward!(input::Array{T1,1}, layer::PermutationLayer, output::Array{T2,1}) where { T1 <: Real, T2 <: Real }

    # fetch variables
    P = getP(layer)

    # determine result
    mul!(input, P', output)
    
end

# jacobian of the additive coupling layer
function _jacobian(layer::PermutationLayer, input::Array{T1,1}) where { T1 <: Real }

    # return result
    return getP(layer)
    
end
jacobian(layer::PermutationLayer, input::Array{T,1}) where { T <: Real } = _jacobian(layer, input)
Broadcast.broadcasted(::typeof(jacobian), layer::PermutationLayer, input::Array{Array{T,1},1}) where { T <: Real } = broadcast(_jacobian, Ref(layer), input)

# inverse jacobian of the additive coupling layer
function _inv_jacobian(layer::PermutationLayer, output::Array{T1,1}) where { T1 <: Real }
    
    # return result
    return getP(layer)'

end
inv_jacobian(layer::PermutationLayer, output::Array{T,1}) where { T <: Real } = _inv_jacobian(layer, output)
Broadcast.broadcasted(::typeof(inv_jacobian), layer::PermutationLayer, output::Array{Array{T,1},1}) where { T <: Real } = broadcast(_inv_jacobian, Ref(layer), output)

# extra utility functions 
det_jacobian(layer::PermutationLayer, input::Array{T,1})           where { T <: Real}   = det(getP(layer))
det_jacobian(layer::PermutationLayer)                              where { T <: Real}   = det(getP(layer))
absdet_jacobian(layer::PermutationLayer, input::Array{T,1})        where { T <: Real}   = 1.0
absdet_jacobian(layer::PermutationLayer)                           where { T <: Real}   = 1.0
logdet_jacobian(layer::PermutationLayer, input::Array{T,1})        where { T <: Real}   = 0.0
logdet_jacobian(layer::PermutationLayer)                           where { T <: Real}   = 0.0
logabsdet_jacobian(layer::PermutationLayer, input::Array{T,1})     where { T <: Real}   = 0.0
logabsdet_jacobian(layer::PermutationLayer)                        where { T <: Real}   = 0.0

detinv_jacobian(layer::PermutationLayer, output::Array{T,1})       where { T <: Real}   = det(getP(layer)')
detinv_jacobian(layer::PermutationLayer)                           where { T <: Real}   = det(getP(layer)')
absdetinv_jacobian(layer::PermutationLayer, output::Array{T,1})    where { T <: Real}   = 1.0
absdetinv_jacobian(layer::PermutationLayer)                        where { T <: Real}   = 1.0
logdetinv_jacobian(layer::PermutationLayer, output::Array{T,1})    where { T <: Real}   = 0.0
logdetinv_jacobian(layer::PermutationLayer)                        where { T <: Real}   = 0.0
logabsdetinv_jacobian(layer::PermutationLayer, output::Array{T,1}) where { T <: Real}   = 0.0
logabsdetinv_jacobian(layer::PermutationLayer)                     where { T <: Real}   = 0.0