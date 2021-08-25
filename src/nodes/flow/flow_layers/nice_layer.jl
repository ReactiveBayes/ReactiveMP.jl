## TODO: expand to arbitrarily sized inputs

export NiceLayer
export getf, getmap
export forward, forward!, backward, backward!
export jacobian, inv_jacobian
export det_jacobian, absdet_jacobian, logdet_jacobian, logabsdet_jacobian
export detinv_jacobian, absdetinv_jacobian, logdetinv_jacobian, logabsdetinv_jacobian

import Base: eltype

@doc raw"""
The NICE layer specifies an invertible function ``{\bf{y}} = g({\bf{x}})`` following the specific structure (for the mapping ``g: \mathbb{R}^2 \rightarrow \mathbb{R}^2``):

```math
    \begin{align}
        y_1 &= x_1 \\
        y_2 &= x_2 + f(x_1)
    \end{align}
```

where ``f(\cdot)`` denotes an arbitrary function with mapping ``f: \mathbb{R} \rightarrow \mathbb{R}``. This function can be chosen arbitrarily complex. Non-linear functions (neural networks) are often chosen to model complex relationships. From the definition of the model, invertibility can be easily achieved as

```math
    \begin{align}
        x_1 &= y_1 \\
        x_2 &= y_2 - f(y_1)
    \end{align}
```

The current implementation only allows for the mapping ``g: \mathbb{R}^2 \rightarrow \mathbb{R}^2``, although this layer can be generalized for arbitrary input dimensions.

`NiceLayer(f <: AbstractNeuralNetwork)` creates the layer structure with function `f`.

### Example
```julia
f = PlanarMap()
layer = NiceLayer(f)
```

This layer structure has been introduced in:

Dinh, Laurent, David Krueger, and Yoshua Bengio. "Nice: Non-linear independent components estimation." _arXiv preprint_ arXiv:1410.8516 (2014).
"""
struct NiceLayer{T <: AbstractNeuralNetwork} <: AbstractFlowLayer
    f       :: T
end

# get-functions for the NiceLayer structure
getf(layer::NiceLayer)     = layer.f
getmap(layer::NiceLayer)   = layer.f

# custom Base function for the NiceLayer structure
eltype(layer::NiceLayer{T})  where { T }              = eltype(T)
eltype(::Type{NiceLayer{T}}) where { T }              = eltype(T)

# forward pass through the NICE layer
function _forward(layer::NiceLayer, input::Array{T,1}) where { T <: Real } 

    # check dimensionality
    @assert length(input) == 2 "The NiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    result = [input[1], input[2] + forward(f, input[1])]

    # return result
    return result
    
end
forward(layer::NiceLayer, input::Array{T,1}) where { T <: Real } = _forward(layer, input)
Broadcast.broadcasted(::typeof(forward), layer::NiceLayer, input::Array{Array{T,1},1}) where { T <: Real } = broadcast(_forward, Ref(layer), input)

# inplace forward pass through the NICE layer
function forward!(output::Array{T1,1}, layer::NiceLayer, input::Array{T2,1}) where { T1 <: Real, T2 <: Real }

    # check dimensionality
    @assert length(input) == 2 "The NiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    output[1] = input[1] 
    output[2] = input[2] 
    output[2] += forward(f, input[1])
    
end

# backward pass through the NICE layer
function _backward(layer::NiceLayer, output::Array{T,1}) where { T <: Real }

    # check dimensionality
    @assert length(output) == 2 "The NiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    result = [output[1], output[2] - forward(f, output[1])]

    # return result
    return result
    
end
backward(layer::NiceLayer, output::Array{T,1}) where { T <: Real } = _backward(layer, output)
Broadcast.broadcasted(::typeof(backward), layer::NiceLayer, output::Array{Array{T,1},1}) where { T <: Real } = broadcast(_backward, Ref(layer), output)

# inplace backward pass through the NICE layer
function backward!(input::Array{T1,1}, layer::NiceLayer, output::Array{T2,1}) where { T1 <: Real, T2 <: Real }

    # check dimensionality
    @assert length(output) == 2 "The NiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    input[1] = output[1]
    input[2] = output[2] - forward(f, output[1])
    
end

# jacobian of the NICE layer
function _jacobian(layer::NiceLayer, input::Array{T1,1}) where { T1 <: Real }

    # check dimensionality
    @assert length(input) == 2 "The NiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # promote type for allocating output
    T = promote_type(eltype(layer), T1)

    # determine result  
    result = zeros(T, 2, 2)
    result[1,1] = 1.0
    result[2,1] = jacobian(f, input[1])
    result[2,2] = 1.0
    
    # return result
    return LowerTriangular(result)
    
end
jacobian(layer::NiceLayer, input::Array{T,1}) where { T <: Real } = _jacobian(layer, input)
Broadcast.broadcasted(::typeof(jacobian), layer::NiceLayer, input::Array{Array{T,1},1}) where { T <: Real } = broadcast(_jacobian, Ref(layer), input)

# inverse jacobian of the NICE layer
function _inv_jacobian(layer::NiceLayer, output::Array{T1,1}) where { T1 <: Real }

    # check dimensionality
    @assert length(output) == 2 "The NiceLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # promote type for allocating output
    T = promote_type(eltype(layer), T1)

    # determine result
    result = zeros(T, 2, 2)
    result[1,1] = 1.0
    result[2,1] = -jacobian(f,output[1])
    result[2,2] = 1.0
    
    # return result
    return LowerTriangular(result)

end
inv_jacobian(layer::NiceLayer, output::Array{T,1}) where { T <: Real } = _inv_jacobian(layer, output)
Broadcast.broadcasted(::typeof(inv_jacobian), layer::NiceLayer, output::Array{Array{T,1},1}) where { T <: Real } = broadcast(_inv_jacobian, Ref(layer), output)

# extra utility functions 
det_jacobian(layer::NiceLayer, input::Array{T,1})           where { T <: Real}   = 1.0
absdet_jacobian(layer::NiceLayer, input::Array{T,1})        where { T <: Real}   = 1.0
logdet_jacobian(layer::NiceLayer, input::Array{T,1})        where { T <: Real}   = 0.0
logabsdet_jacobian(layer::NiceLayer, input::Array{T,1})     where { T <: Real}   = 0.0

detinv_jacobian(layer::NiceLayer, output::Array{T,1})       where { T <: Real}   = 1.0
absdetinv_jacobian(layer::NiceLayer, output::Array{T,1})    where { T <: Real}   = 1.0
logdetinv_jacobian(layer::NiceLayer, output::Array{T,1})    where { T <: Real}   = 0.0
logabsdetinv_jacobian(layer::NiceLayer, output::Array{T,1}) where { T <: Real}   = 0.0