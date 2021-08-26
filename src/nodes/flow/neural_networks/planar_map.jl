export PlanarMap
export getu, getw, getb, getall, setu!, setw!, setb!
export forward, forward!
export jacobian, jacobian!, inv_jacobian, det_jacobian, absdet_jacobian, logabsdet_jacobian

import Base: eltype, size, length

@doc raw"""
The PlanarMap function is defined as

```math
f({\bf{x}}) = {\bf{x}} + {\bf{u}} \tanh({\bf{w}}^\top {\bf{x}} + b)
```

with input and output dimension ``D``. Here ``{\bf{x}}\in \mathbb{R}^D`` represents the input of the function. Furthermore ``{\bf{u}}\in \mathbb{R}^D``, ``{\bf{w}}\in \mathbb{R}^D`` and ``b\in\mathbb{R}`` represent the parameters of the function. The function contracts and expands the input space. 

This function has been introduced in:

Rezende, Danilo, and Shakir Mohamed. "Variational inference with normalizing flows." _International conference on machine learning._ PMLR, 2015.
"""
mutable struct PlanarMap{T1, T2 <: Real} <: AbstractNeuralNetwork
    u       :: T1
    w       :: T1
    b       :: T2
    function PlanarMap(u::T1, w::T1, b::T2) where { T1, T2 <: Real}
        @assert length(u) == length(w) "The parameters u and w in the PlanarMap structure should have equal length."
        return new{T1,T2}(u, w, float(b))
    end
end

@doc raw"""
The `PlanarMap(dim::Int64)` function creates a mutable `PlanarMap` structure with parameters corresponding to input of dimensions `dim`. The parameters are each random sampled from a standard (multivariate) normal distribution.
"""
function PlanarMap(dim::Int64)
    return PlanarMap(randn(dim), randn(dim), randn())
end

@doc raw"""
The `PlanarMap()` function creates a mutable `PlanarMap` structure with parameters corresponding to input of dimension 1. The parameters are each random sampled from a standard normal distribution.
"""
function PlanarMap()
    return PlanarMap(randn(), randn(), randn())
end

# get-functions for the PlanarMap structure.
getu(f::PlanarMap)              = return f.u
getw(f::PlanarMap)              = return f.w
getb(f::PlanarMap)              = return f.b
getall(f::PlanarMap)            = return f.u, f.w, f.b

# set-functions for the PlanarMap structure
function setu!(f::PlanarMap{T1,T2}, u::T1) where { T1, T2 <: Real}
    @assert length(f.u) == length(u) "The dimensionality of the current value of u and its new value do not match."
    f.u = u
end

function setw!(f::PlanarMap{T1,T2}, w::T1) where { T1, T2 <: Real}
    @assert length(f.w) == length(w) "The dimensionality of the current value of w and its new value do not match."
    f.w = w
end

function setb!(f::PlanarMap{T1,T2}, b::T2) where { T1, T2 <: Real }
    f.b = b
end

# custom Base function for the PlanarMap structure
eltype(f::PlanarMap{T1,T2}) where { T1 <: Real, T2 <: Real}                 = promote_type(T1, T2)
eltype(f::PlanarMap{T1,T2}) where { T1 <: AbstractArray, T2 <: Real}        = promote_type(eltype(T1), T2)
eltype(::Type{PlanarMap{T1,T2}}) where { T1 <: Real, T2 <: Real}            = promote_type(T1, T2)
eltype(::Type{PlanarMap{T1,T2}}) where { T1 <: AbstractArray, T2 <: Real}   = promote_type(eltype(T1), T2)

size(f::PlanarMap{T1,T2}) where { T1 <: Real, T2 <: Real}                   = 1
size(f::PlanarMap{T1,T2}) where { T1 <: AbstractArray, T2 <: Real}          = length(f.u)

length(f::PlanarMap{T1,T2}) where { T1 <: Real, T2 <: Real}                 = 1
length(f::PlanarMap{T1,T2}) where { T1 <: AbstractArray, T2 <: Real}        = length(f.u)

# forward pass through the PlanarMap function (multivariate input)
function _forward(f::PlanarMap{T1,T2}, input::T1) where { T1, T2 <: Real }

    # fetch values
    u, w, b = getall(f)
    
    # calculate result
    result = zeros(size(u))
    result .*= tanh(dot(w, input) + b) 
    result .+= input

    # return result
    return result

end
forward(f::PlanarMap{T1,T2}, input::T1) where { T1, T2 <: Real } = _forward(f, input)
Broadcast.broadcasted(::typeof(forward), f::PlanarMap{T1,T2}, input::Array{T1,1}) where { T1, T2 <: Real } = broadcast(_forward, Ref(f), input)


# forward pass through the PlanarMap function (univariate input)
function _forward(f::PlanarMap{T1,T2}, input::T3) where { T1 <: Real, T2 <: Real, T3 <: Real }

    # fetch values
    u, w, b = getall(f)
    
    # calculate result
    result = copy(u)
    result *= tanh(dot(w, input) + b)
    result += input

    # return result
    return result

end
forward(f::PlanarMap{T1,T2}, input::T3) where { T1 <: Real, T2 <: Real, T3 <: Real } = _forward(f, input)
Broadcast.broadcasted(::typeof(forward), f::PlanarMap{T1,T2}, input::Array{T3,1}) where { T1 <: Real, T2 <: Real, T3 <: Real } = broadcast(_forward, Ref(f), input)

# inplace forward pass through the PlanarMap function (multivariate input)
function forward!(output::T1, f::PlanarMap{T1,T2}, input::T1) where { T1, T2 <: Real }

    # check dimensionality
    @assert length(output) == length(input) "The length of the preallocated vector does not seem to match the length of the input vector."

    # fetch values
    u, w, b = getall(f)
    
    # calculate result
    output .= u
    output .*= tanh(dot(w, input) + b) 
    output .+= input

end

# jacobian of the PlanarMap function (multivariate input)
function _jacobian(f::PlanarMap{T1,T2}, input::T1) where { T1, T2 <: Real}

    # fetch values 
    u, w, b = getall(f)

    # calculate result
    result = u*w'
    result .*= dtanh(dot(w, input) + b)
    @inbounds for k = 1:length(input)
        result[k,k] += 1.0
    end

    # return result
    return result

end
jacobian(f::PlanarMap{T1,T2}, input::T1) where { T1, T2 <: Real } = _jacobian(f, input)
Broadcast.broadcasted(::typeof(jacobian), f::PlanarMap{T1,T2}, input::Array{T1,1}) where { T1, T2 <: Real } = broadcast(_jacobian, Ref(f), input)

# jacobian of the PlanarMap function (univariate input)
function _jacobian(f::PlanarMap{T1,T2}, input::T3) where { T1 <: Real, T2 <: Real, T3 <: Real } 

    # fetch values 
    u, w, b = getall(f)

    # calculate result (optimized)
    result = u * w * dtanh(w * input + b) + 1

    # return result
    return result

end
jacobian(f::PlanarMap{T1,T2}, input::T3) where { T1 <: Real, T2 <: Real, T3 <: Real } = _jacobian(f, input)
Broadcast.broadcasted(::typeof(jacobian), f::PlanarMap{T1,T2}, input::Array{T3,1}) where { T1 <: Real, T2 <: Real, T3 <: Real } = broadcast(_jacobian, Ref(f), input)

# inplace jacobian of the PlanarMap function (multivariate input)
function jacobian!(output::Array{T2,2}, f::PlanarMap{T1,T2}, input::T1) where { T1, T2 <: Real}

    # check whether the dimensionality is correct
    @assert size(output) == (length(input), length(f.u)) "The dimensionality of the preallocated jacobian matrix seems incorrect."

    # fetch values 
    u, w, b = getall(f)

    # calculate result
    for ku = 1:length(u)
        for kw = 1:length(w)
            output[ku,kw] = u[ku]*w[kw]
        end
    end
    output .*= dtanh(dot(w, input) + b)
    @inbounds for k = 1:length(input)
        output[k,k] += 1.0
    end

end

# determinant of the jacobian of the PlanarMap function (multivariate input)
det_jacobian(f::PlanarMap{T1,T2}, input::T1) where { T1, T2 <: Real} = det(jacobian(f, input))

# determinant of the jacobian of the PlanarMap function (univariate input)
function det_jacobian(f::PlanarMap{T, T}, input::T) where { T <: Real }

    # fetch values
    u, w, b = getall(f)

    # calculate result
    result = 1 + dot(u, w)*dtanh(dot(w, input) + b)

    # return result
    return result

end

# extra utility function (multivariate)
inv_jacobian(f::PlanarMap{T1,T2}, input::T1) where { T1, T2 <: Real }       = inv(jacobian(f, input))
absdet_jacobian(f::PlanarMap{T1,T2}, input::T1) where { T1, T2 <: Real }    = abs(det_jacobian(f, input))
logabsdet_jacobian(f::PlanarMap{T1,T2}, input::T1) where { T1, T2 <: Real}  = logabsdet(jacobian(f, input))

# extra utility functions (univariate)
inv_jacobian(f::PlanarMap{T,T}, input::T) where { T <: Real }               = 1.0 / jacobian(f, input)
absdet_jacobian(f::PlanarMap{T,T}, input::T) where { T <: Real }            = abs(det_jacobian(f, input))
logabsdet_jacobian(f::PlanarMap{T,T}, input::T) where { T <: Real}          = log(absdet_jacobian(f, input))