## TODO: expand to arbitrarily sized inputs

export ReverseNiceLayer
export getf, getmap
export forward, forward!, backward, backward!
export jacobian, inv_jacobian
export det_jacobian, absdet_jacobian, logdet_jacobian, logabsdet_jacobian
export detinv_jacobian, absdetinv_jacobian, logdetinv_jacobian, logabsdetinv_jacobian

import Base: eltype

@doc raw"""
The (reversed) NICE layer specifies an invertible function ``{\bf{y}} = g({\bf{x}})`` following the specific structure (for the mapping ``g: \mathbb{R}^2 \rightarrow \mathbb{R}^2``):

```math
    \begin{align}
        y_1 &= x_1 + f(x_2)\\
        y_2 &= x_2
    \end{align}
```

where ``f(\cdot)`` denotes an arbitrary function with mapping ``f: \mathbb{R} \rightarrow \mathbb{R}``. This function can be chosen arbitrarily complex. Non-linear functions (neural networks) are often chosen to model complex relationships. From the definition of the model, invertibility can be easily achieved as

```math
    \begin{align}
        x_1 &= y_1 - f(y_2)\\
        x_2 &= y_2 
    \end{align}
```

The current implementation only allows for the mapping ``g: \mathbb{R}^2 \rightarrow \mathbb{R}^2``, although this layer can be generalized for arbitrary input dimensions.

`ReverseNiceLayer(f <: AbstractNeuralNetwork)` creates the layer structure with function `f`.

### Example
```julia
f = PlanarMap()
layer = ReverseNiceLayer(f)
```

This layer structure has been introduced in:

Dinh, Laurent, David Krueger, and Yoshua Bengio. "Nice: Non-linear independent components estimation." _arXiv preprint_ arXiv:1410.8516 (2014).
"""
struct ReverseNiceLayer{T <: AbstractNeuralNetwork} <: AbstractFlowLayer
    f       :: T
end

# ReverseNiceLayer methods
getf(layer::ReverseNiceLayer)              = layer.f
getmap(layer::ReverseNiceLayer)            = layer.f

# custom Base function for the ReverseNiceLayer structure
eltype(layer::ReverseNiceLayer{T})  where { T }      = eltype(T)
eltype(::Type{ReverseNiceLayer{T}}) where { T }      = eltype(T)

# forward pass through the reverse NICE layer
function _forward(layer::ReverseNiceLayer, input::Array{T,1}) where { T <: Real } 

    # check dimensionality
    @assert length(input) == 2 "The ReverseNiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    result = [input[1] + forward(f, input[2]), input[2]]

    # return result
    return result
    
end
forward(layer::ReverseNiceLayer, input::Array{T,1}) where { T <: Real } = _forward(layer, input)
Broadcast.broadcasted(::typeof(forward), layer::ReverseNiceLayer, input::Array{T,1}) where { T <: Real } = broadcast(_forward, Ref(layer), input)

# inplace forward pass through the reverse NICE layer
function forward!(output::Array{T1,1}, layer::ReverseNiceLayer, input::Array{T2,1}) where { T1 <: Real, T2 <: Real }

    # check dimensionality
    @assert length(input) == 2 "The ReverseNiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    output[1] = input[1] 
    output[2] = input[2] 
    output[1] += forward(f, input[2])
    
end

# backward pass through the reverse NICE layer
function _backward(layer::ReverseNiceLayer, output::Array{T,1}) where { T <: Real }

    # check dimensionality
    @assert length(output) == 2 "The ReverseNiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    result = [output[1] - forward(f, output[2]), output[2]]

    # return result
    return result
    
end
backward(layer::ReverseNiceLayer, output::Array{T,1}) where { T <: Real } = _backward(layer, input)
Broadcast.broadcasted(::typeof(backward), layer::ReverseNiceLayer, output::Array{T,1}) where { T <: Real } = broadcast(_backward, Ref(layer), output)

# inplace backward pass through the reverse, NICE layer
function backward!(input::Array{T1,1}, layer::ReverseNiceLayer, output::Array{T2,1}) where { T1 <: Real, T2 <: Real }

    # check dimensionality
    @assert length(output) == 2 "The ReverseNiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    input[1] = output[1] - forward(f, output[2])
    input[2] = output[2]
    
end

# jacobian of the reverse NICE layer
function _jacobian(layer::ReverseNiceLayer, input::Array{T1,1}) where { T1 <: Real }

    # check dimensionality
    @assert length(input) == 2 "The ReverseNiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # promote type for allocating output
    T = promote_type(eltype(layer), T1)

    # determine result  
    result = zeros(T, 2, 2)
    result[1,1] = 1.0
    result[1,2] = jacobian(f, input[1])
    result[2,2] = 1.0
    
    # return result
    return UpperTriangular(result)
    
end
jacobian(layer::ReverseNiceLayer, input::Array{T,1}) where { T <: Real } = _jacobian(layer, input)
Broadcast.broadcasted(::typeof(jacobian), layer::ReverseNiceLayer, input::Array{T,1}) where { T <: Real } = broadcast(_jacobian, Ref(layer), input)

# inverse jacobian of the reverse NICE layer
function _inv_jacobian(layer::ReverseNiceLayer, output::Array{T1,1}) where { T1 <: Real }

    # check dimensionality
    @assert length(output) == 2 "The ReverseNiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # promote type for allocating output
    T = promote_type(eltype(layer), T1)

    # determine result
    result = zeros(T, 2, 2)
    result[1,1] = 1.0
    result[1,2] = -jacobian(f,output[1])
    result[2,2] = 1.0
    
    # return result
    return UpperTriangular(result)

end
inv_jacobian(layer::ReverseNiceLayer, output::Array{T,1}) where { T <: Real } = _inv_jacobian(layer, input)
Broadcast.broadcasted(::typeof(inv_jacobian), layer::ReverseNiceLayer, output::Array{T,1}) where { T <: Real } = broadcast(_inv_jacobian, Ref(layer), output)

# extra utility functions 
det_jacobian(layer::ReverseNiceLayer, input::Array{T,1})            where { T <: Real } = 1
absdet_jacobian(layer::ReverseNiceLayer, input::Array{T,1})         where { T <: Real } = 1
logdet_jacobian(layer::ReverseNiceLayer, input::Array{T,1})         where { T <: Real } = 0
logabsdet_jacobian(layer::ReverseNiceLayer, input::Array{T,1})      where { T <: Real } = 0

detinv_jacobian(layer::ReverseNiceLayer, output::Array{T,1})        where { T <: Real } = 1
absdetinv_jacobian(layer::ReverseNiceLayer, output::Array{T,1})     where { T <: Real } = 1
logdetinv_jacobian(layer::ReverseNiceLayer, output::Array{T,1})     where { T <: Real } = 0
logabsdetinv_jacobian(layer::ReverseNiceLayer, output::Array{T,1})  where { T <: Real } = 0