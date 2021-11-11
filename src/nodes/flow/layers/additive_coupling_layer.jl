## TODO: fix for non-unity partition dimensions

export AdditiveCouplingLayer
export getf, getflow, getdim
export forward, forward!, backward, backward!
export jacobian, inv_jacobian
export det_jacobian, absdet_jacobian, logdet_jacobian, logabsdet_jacobian
export detinv_jacobian, absdetinv_jacobian, logdetinv_jacobian, logabsdetinv_jacobian

import Base: eltype

@doc raw"""
The additive coupling layer specifies an invertible function ``{\bf{y}} = g({\bf{x}})`` following the specific structure (for the mapping ``g: \mathbb{R}^2 \rightarrow \mathbb{R}^2``):

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

`AdditiveCouplingLayer(f <: AbstractCouplingFlow)` creates the layer structure with function `f`.

### Example
```julia
f = PlanarFlow()
layer = AdditiveCouplingLayer(f)
```

This layer structure has been introduced in:

Dinh, Laurent, David Krueger, and Yoshua Bengio. "Nice: Non-linear independent components estimation." _arXiv preprint_ arXiv:1410.8516 (2014).
"""
struct AdditiveCouplingLayer{ T <: NTuple{N,AbstractCouplingFlow} where { N } } <: AbstractCouplingLayer
    dim           :: Int
    f             :: T
    partition_dim :: Int
end
struct AdditiveCouplingLayerEmpty{ T <: NTuple{N,AbstractCouplingFlowEmpty} where { N } } <: AbstractCouplingLayer
    dim           :: Int
    f             :: T
    partition_dim :: Int
end
struct AdditiveCouplingLayerPlaceholder{T <: AbstractCouplingFlowEmpty, B } <: AbstractLayerPlaceholder
    f             :: T
    partition_dim :: Int
    permute       :: Val{B}
end
function AdditiveCouplingLayer(flow::T; partition_dim::Int=1, permute::Bool=true) where { T <: AbstractCouplingFlowPlaceholder }
    return AdditiveCouplingLayerPlaceholder(prepare(partition_dim, flow), partition_dim, Val(permute))
end

# include permute as value type for type stability
function _prepare(dim::Int, layer::AdditiveCouplingLayerPlaceholder{T, true}) where { T } 
    ## TODO: generalize for non-1 partition dim and overlap
    @assert dim % getpartitiondim(layer) == 0 "The input dimensionality is not exactly divisible by the partition dimension."
    nr_maps = ( dim ÷ getpartitiondim(layer) ) - 1
    maps = ntuple((x) -> getf(layer), Val(nr_maps))
    return AdditiveCouplingLayerEmpty(dim, maps, getpartitiondim(layer)), PermutationLayer(dim)
end
function _prepare(dim::Int, layer::AdditiveCouplingLayerPlaceholder{T, false}) where { T } 
    ## TODO: generalize for non-1 partition dim and overlap
    @assert dim % getpartitiondim(layer) == 0 "The input dimensionality is not exactly divisible by the partition dimension."
    nr_maps = ( dim ÷ getpartitiondim(layer) ) - 1
    maps = ntuple((x) -> getf(layer), Val(nr_maps))
    return AdditiveCouplingLayerEmpty(dim, maps, getpartitiondim(layer))
end

# compile layer (tuples are compiled according to compile in flow_model.jl)
compile(layer::AdditiveCouplingLayerEmpty)          = AdditiveCouplingLayer(getdim(layer), compile(getf(layer)), getpartitiondim(layer))
compile(layer::AdditiveCouplingLayerEmpty, params)  = AdditiveCouplingLayer(getdim(layer), compile(getf(layer), params), getpartitiondim(layer))

# calculates the number of parameters in the model
nr_params(layer::AdditiveCouplingLayer)             = mapreduce(nr_params, +, getf(layer))
nr_params(layer::AdditiveCouplingLayerEmpty)        = mapreduce(nr_params, +, getf(layer))

# get-functions for the AdditiveCouplingLayer structure
getf(layer::AdditiveCouplingLayer)              = layer.f
getflow(layer::AdditiveCouplingLayer)           = layer.f
getdim(layer::AdditiveCouplingLayer)            = layer.dim
getpartitiondim(layer::AdditiveCouplingLayer)   = layer.partition_dim

# get-functions for the AdditiveCouplingLayerPlaceholder structure
getf(layer::AdditiveCouplingLayerPlaceholder)               = layer.f
getflow(layer::AdditiveCouplingLayerPlaceholder)            = layer.f
getpartitiondim(layer::AdditiveCouplingLayerPlaceholder)    = layer.partition_dim

# get-functions for the AdditiveCouplingLayerEmpty structure
getdim(layer::AdditiveCouplingLayerEmpty)           = layer.dim
getf(layer::AdditiveCouplingLayerEmpty)             = layer.f
getflow(layer::AdditiveCouplingLayerEmpty)          = layer.f
getpartitiondim(layer::AdditiveCouplingLayerEmpty)  = layer.partition_dim

# custom Base function for the AdditiveCouplingLayer structure
eltype(layer::AdditiveCouplingLayer{T})  where { T }              = promote_type(map(eltype, getf(layer))...)

# forward pass through the additive coupling layer
function _forward(layer::AdditiveCouplingLayer, input::AbstractVector{ <: Real }) 

    # allocate result
    result = similar(input)

    # calculate result
    forward!(result, layer, input)

    # return result
    return result
    
end
forward(layer::AdditiveCouplingLayer, input::AbstractVector{ <: Real }) = _forward(layer, input)
Broadcast.broadcasted(::typeof(forward), layer::AdditiveCouplingLayer, input::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_forward, Ref(layer), input)

# inplace forward pass through the additive coupling layer
function forward!(output::AbstractVector{ <: Real }, layer::AdditiveCouplingLayer, input::AbstractVector{ <: Real })

    # fetch variables
    f = getf(layer)
    dim = getdim(layer)
    pdim = getpartitiondim(layer)

    # check dimensionality
    @assert length(input) == dim "The dimensionality of the AdditiveCouplingLayer does not correspond to the length of the passed input/output."

    # optimized version for scalar partition dimension
    if pdim == 1 
        output[1] = input[1]
        for k = 2:dim
            output[k] = input[k]
            output[k] += forward(f[k-1], input[k-1])
        end
    else
        view(output, 1:pdim) .= view(input, 1:pdim) 
        for k = 2:dim÷pdim
            view(output, 1+(k-1)*pdim:k*pdim) .= view(input, 1+(k-1)*pdim:k*pdim)
            view(output, 1+(k-1)*pdim:k*pdim) .+= forward(f[k-1], view(input, 1+(k-2)*pdim:(k-1)*pdim))
        end
    end
    
end

# backward pass through the additive coupling layer
function _backward(layer::AdditiveCouplingLayer, output::AbstractVector{ <: Real })

    # allocate result
    result = similar(output)

    # calculate result
    backward!(result, layer, output)

    # return result
    return result
    
end
backward(layer::AdditiveCouplingLayer, output::AbstractVector{ <: Real }) = _backward(layer, output)
Broadcast.broadcasted(::typeof(backward), layer::AdditiveCouplingLayer, output::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_backward, Ref(layer), output)

# inplace backward pass through the additive coupling layer
function backward!(input::AbstractVector{ <: Real }, layer::AdditiveCouplingLayer, output::AbstractVector{ <: Real })

    # fetch variables
    f = getf(layer)
    dim = getdim(layer)
    pdim = getpartitiondim(layer)

    # check dimensionality
    @assert length(input) == dim "The dimensionality of the AdditiveCouplingLayer does not correspond to the length of the passed input/output."

    # determine result
    if pdim == 1
        input[1] = output[1]
        for k = 2:dim
            input[k] = output[k]
            input[k] -= forward(f[k-1], input[k-1])
        end
    else
        view(input, 1:pdim) .= view(output, 1:pdim) 
        for k = 2:dim÷pdim
            view(input, 1+(k-1)*pdim:k*pdim) .= view(output, 1+(k-1)*pdim:k*pdim)
            view(input, 1+(k-1)*pdim:k*pdim) .-= forward(f[k-1], view(input, 1+(k-2)*pdim:(k-1)*pdim))
        end
    end
    
end

# jacobian of the additive coupling layer
function _jacobian(layer::AdditiveCouplingLayer, input::AbstractVector{T}) where { T <: Real }

    # fetch variables
    dim = getdim(layer)

    # allocate jacobian
    Ti = promote_type(eltype(layer), T)
    result = zeros(Ti, dim, dim)

    # determine result  
    jacobian!(result, layer, input)
    
    # return result
    return LowerTriangular(result)
    
end
jacobian(layer::AdditiveCouplingLayer, input::AbstractVector{ <: Real }) = _jacobian(layer, input)
Broadcast.broadcasted(::typeof(jacobian), layer::AdditiveCouplingLayer, input::AbstractVector{ AbstractVector{ <: Real } }) = broadcast(_jacobian, Ref(layer), input)

# inplace jacobian through the additive coupling layer
function jacobian!(result::AbstractMatrix{T}, layer::AdditiveCouplingLayer, input::AbstractVector{ <: Real }) where { T <: Real }

    # fetch variables
    f = getf(layer)
    dim = getdim(layer)
    pdim = getpartitiondim(layer)

    # check dimensionality
    @assert length(input) == dim "The dimensionality of the AdditiveCouplingLayer does not correspond to the length of the passed input/output."

    # determine result
    result .= zero(T)
    for k = 1:dim÷pdim
        result[k,k] = one(T)
    end
    for k = 1:dim÷pdim-1
        result[1+k*pdim:(k+1)*pdim, 1+(k-1)*pdim:k*pdim] .+= jacobian(f[k], input[1+(k-1)*pdim:k*pdim])
    end
    
end


# inverse jacobian of the additive coupling layer
function _inv_jacobian(layer::AdditiveCouplingLayer, output::AbstractVector{T}) where { T <: Real }

    # fetch variables
    dim = getdim(layer)

    # allocate jacobian
    Ti = promote_type(eltype(layer), T)
    result = zeros(Ti, dim, dim)

    # determine result  
    inv_jacobian!(result, layer, output)
    
    # return result
    return LowerTriangular(result)

end
inv_jacobian(layer::AdditiveCouplingLayer, output::AbstractVector{ <: Real }) = _inv_jacobian(layer, output)
Broadcast.broadcasted(::typeof(inv_jacobian), layer::AdditiveCouplingLayer, output::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_inv_jacobian, Ref(layer), output)

# inplace inv_jacobian through the additive coupling layer
function inv_jacobian!(result::AbstractVector{T}, layer::AdditiveCouplingLayer, output::AbstractVector{ <: Real }) where { T <: Real }

    # fetch variables
    f = getf(layer)
    dim = getdim(layer)
    pdim = getpartitiondim(layer)

    # check dimensionality
    @assert length(output) == dim "The dimensionality of the AdditiveCouplingLayer does not correspond to the length of the passed input/output."

    # calculate input of layer for simpler jacobian calculation
    input = backward(layer, output)

    # determine result
    result .= zero(T)
    for k = 1:dim÷pdim
        result[k:end,k] .= one(T)
    end
    for k = 1:dim÷pdim-1
        result[k+1:end, 1:k] .*= -jacobian(f[k], input[1+(k-1)*pdim:k*pdim])
    end
    
end


# extra utility functions 
det_jacobian(layer::AdditiveCouplingLayer, input::AbstractVector{ <: Real })           = 1.0
absdet_jacobian(layer::AdditiveCouplingLayer, input::AbstractVector{ <: Real })        = 1.0
logdet_jacobian(layer::AdditiveCouplingLayer, input::AbstractVector{ <: Real })        = 0.0
logabsdet_jacobian(layer::AdditiveCouplingLayer, input::AbstractVector{ <: Real })     = 0.0

detinv_jacobian(layer::AdditiveCouplingLayer, output::AbstractVector{ <: Real })       = 1.0
absdetinv_jacobian(layer::AdditiveCouplingLayer, output::AbstractVector{ <: Real })    = 1.0
logdetinv_jacobian(layer::AdditiveCouplingLayer, output::AbstractVector{ <: Real })    = 0.0
logabsdetinv_jacobian(layer::AdditiveCouplingLayer, output::AbstractVector{ <: Real }) = 0.0