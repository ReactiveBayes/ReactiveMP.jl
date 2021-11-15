export PermutationLayer

import Base: eltype

@doc raw"""
The permutation layer specifies an invertible mapping ``{\bf{y}} = g({\bf{x}}) = P{\bf{x}}`` where ``P`` is a permutation matrix.
"""
struct PermutationLayer{ T } <: AbstractLayer
    dim :: Int
    P   :: PermutationMatrix{T}
end

function PermutationLayer(dim::T) where { T <: Integer }

    # create random permutation matrix
    P = PermutationMatrix(dim)

    # return layer
    return PermutationLayer(dim, P)

end
function PermutationLayer(P::PermutationMatrix)

    # create random permutation matrix
    @assert size(P,1) == size(P,2) "The passed permutation matrix is not square." 

    # return layer
    return PermutationLayer(size(P,1), P)

end
struct PermutationLayerPlaceholder <: AbstractLayerPlaceholder end
@doc raw"""
`PermutationLayer()` creates a layer that randomly shuffles its input values. The corresponding permutation matrix and its dimensionality are (randomly) generated during model creation.
"""
PermutationLayer() = PermutationLayerPlaceholder() # the function creates a placeholder, of which the dimensionality is set later on.

# prepare placeholder 
_prepare(dim::Int, layer::PermutationLayerPlaceholder) = (PermutationLayer(dim), )
function _prepare(dim::Int, layer::PermutationLayer)
    @assert dim == size(getP(layer),1) == size(getP(layer),2) "The size of the passed permutation matrix does not comply with the dimensionality of the input."
    return (layer, )
end

# compile layer
compile(layer::PermutationLayer, params) = throw(ArgumentError("The permutation matrix does not have any parameters."))
compile(layer::PermutationLayer)         = layer

# fetch number of parameters of layer
nr_params(layer::PermutationLayer) = 0

# get-functions for the PermutationLayer structure
getP(layer::PermutationLayer)     = layer.P
getmat(layer::PermutationLayer)   = layer.P
getdim(layer::PermutationLayer)   = layer.dim

# custom Base function for the PermutationLayer structure
eltype(layer::PermutationLayer{T})  where { T }              = eltype(T)
eltype(::Type{PermutationLayer{T}}) where { T }              = eltype(T)

# forward pass through the permutation layer
function _forward(layer::PermutationLayer, input::AbstractVector{ <: Real })

    # fetch variables
    P = getP(layer)

    # determine result
    result = P*input

    # return result
    return result
    
end
forward(layer::PermutationLayer, input::AbstractVector{ <: Real }) = _forward(layer, input)
Broadcast.broadcasted(::typeof(forward), layer::PermutationLayer, input::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_forward, Ref(layer), input)

# inplace forward pass through the permutation layer
function forward!(output::AbstractVector{ <: Real }, layer::PermutationLayer, input::AbstractVector{ <: Real })

    # fetch variables
    P = getP(layer)

    # determine result
    mul!(output, P, input)
    
end

# backward pass through the permutation layer
function _backward(layer::PermutationLayer, output::AbstractVector{ <: Real })

    # fetch variables
    P = getP(layer)

    # determine result
    result = P'*output

    # return result
    return result
    
end
backward(layer::PermutationLayer, output::AbstractVector{ <: Real }) = _backward(layer, output)
Broadcast.broadcasted(::typeof(backward), layer::PermutationLayer, output::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_backward, Ref(layer), output)

# inplace backward pass through the additive coupling layer
function backward!(input::AbstractVector{ <: Real }, layer::PermutationLayer, output::AbstractVector{ <: Real })

    # fetch variables
    P = getP(layer)

    # determine result
    mul!(input, P', output)
    
end

# jacobian of the additive coupling layer
function _jacobian(layer::PermutationLayer, input::AbstractVector{ <: Real })

    # return result
    return getP(layer)
    
end
jacobian(layer::PermutationLayer, input::AbstractVector{ <: Real }) = _jacobian(layer, input)
Broadcast.broadcasted(::typeof(jacobian), layer::PermutationLayer, input::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_jacobian, Ref(layer), input)

# inverse jacobian of the additive coupling layer
function _inv_jacobian(layer::PermutationLayer, output::AbstractVector{ <: Real })
    
    # return result
    return getP(layer)'

end
inv_jacobian(layer::PermutationLayer, output::AbstractVector{ <: Real }) = _inv_jacobian(layer, output)
Broadcast.broadcasted(::typeof(inv_jacobian), layer::PermutationLayer, output::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_inv_jacobian, Ref(layer), output)

# extra utility functions 
det_jacobian(layer::PermutationLayer, input::AbstractVector{ <: Real })           = det(getP(layer))
det_jacobian(layer::PermutationLayer)                                             = det(getP(layer))
absdet_jacobian(layer::PermutationLayer, input::AbstractVector{ <: Real })        = 1.0
absdet_jacobian(layer::PermutationLayer)                                          = 1.0
logdet_jacobian(layer::PermutationLayer, input::AbstractVector{ <: Real })        = 0.0
logdet_jacobian(layer::PermutationLayer)                                          = 0.0
logabsdet_jacobian(layer::PermutationLayer, input::AbstractVector{ <: Real })     = 0.0
logabsdet_jacobian(layer::PermutationLayer)                                       = 0.0

detinv_jacobian(layer::PermutationLayer, output::AbstractVector{ <: Real })       = det(getP(layer)')
detinv_jacobian(layer::PermutationLayer)                                          = det(getP(layer)')
absdetinv_jacobian(layer::PermutationLayer, output::AbstractVector{ <: Real })    = 1.0
absdetinv_jacobian(layer::PermutationLayer)                                       = 1.0
logdetinv_jacobian(layer::PermutationLayer, output::AbstractVector{ <: Real })    = 0.0
logdetinv_jacobian(layer::PermutationLayer)                                       = 0.0
logabsdetinv_jacobian(layer::PermutationLayer, output::AbstractVector{ <: Real }) = 0.0
logabsdetinv_jacobian(layer::PermutationLayer)                                    = 0.0