export RadialMap
export getz0, getα, getβ, getall, setz0!, setα!, setβ!
export forward, forward!
export jacobian, jacobian!, inv_jacobian, det_jacobian, absdet_jacobian, logabsdet_jacobian

import Base: eltype, size, length

@doc raw"""
The RadialMap function is defined as

```math
f({\bf{x}}) = {\bf{x}} + \frac{\beta({bf{z}} - {\bf{z}}_0)}{\alpha + |{\bf{z}} - {\bf{z}}_0|}
```

with input and output dimension ``D``. Here ``{\bf{x}}\in \mathbb{R}^D`` represents the input of the function. Furthermore ``{\bf{z}}_0\in \mathbb{R}^D``, ``\alpha\in \mathbb{R}`` and ``\beta\in\mathbb{R}`` represent the parameters of the function. The function contracts and expands the input space. 

This function has been introduced in:

Rezende, Danilo, and Shakir Mohamed. "Variational inference with normalizing flows." _International conference on machine learning._ PMLR, 2015.
"""
mutable struct RadialMap{T1, T2 <: Real} <: AbstractNeuralNetwork
    z0      :: T1
    α       :: T2
    β       :: T2
    function RadialMap(z0::T1, α::T2, β::T2) where { T1, T2 <: Real}
        @assert α > 0
        return new{T1,T2}(z0, float(α), float(β))
    end
end

@doc raw"""
The `RadialMap(dim::Int64)` function creates a mutable `RadialMap` structure with parameters corresponding to input of dimensions `dim`. The parameters are each random sampled from a standard (multivariate) normal distribution.
"""
function RadialMap(dim::Int64)
    return RadialMap(randn(dim), rand(), randn())
end

@doc raw"""
The `RadialMap()` function creates a mutable `RadialMap` structure with parameters corresponding to input of dimension 1. The parameters are each random sampled from a standard normal distribution.
"""
function RadialMap()
    return RadialMap(randn(), rand(), randn())
end

# get-functions for the RadialMap structure.
getz0(f::RadialMap)             = return f.z0
getα(f::RadialMap)              = return f.α
getβ(f::RadialMap)              = return f.β
getall(f::RadialMap)            = return f.z0, f.α, f.β

# set-functions for the RadialMap structure
function setz0!(f::RadialMap{T1,T2}, z0::T1) where { T1, T2 <: Real}
    @assert length(f.z0) == length(z0) "The dimensionality of the current value of z0 and its new value do not match."
    f.z0 = z0
end

function setα!(f::RadialMap{T1,T2}, α::T2) where { T1, T2 <: Real}
    f.α = α
end

function setβ!(f::RadialMap{T1,T2}, β::T2) where { T1, T2 <: Real }
    f.β = β
end

# custom Base function for the RadialMap structure
eltype(f::RadialMap{T1,T2}) where { T1 <: Real, T2 <: Real}                 = promote_type(T1, T2)
eltype(f::RadialMap{T1,T2}) where { T1 <: AbstractArray, T2 <: Real}        = promote_type(eltype(T1), T2)
eltype(::Type{RadialMap{T1,T2}}) where { T1 <: Real, T2 <: Real}            = promote_type(T1, T2)
eltype(::Type{RadialMap{T1,T2}}) where { T1 <: AbstractArray, T2 <: Real}   = promote_type(eltype(T1), T2)

size(f::RadialMap{T1,T2}) where { T1 <: Real, T2 <: Real}                   = 1
size(f::RadialMap{T1,T2}) where { T1 <: AbstractArray, T2 <: Real}          = length(f.z0)

length(f::RadialMap{T1,T2}) where { T1 <: Real, T2 <: Real}                 = 1
length(f::RadialMap{T1,T2}) where { T1 <: AbstractArray, T2 <: Real}        = length(f.z0)

# forward pass through the RadialMap function (multivariate input)
function _forward(f::RadialMap{T1,T2}, input::T1) where { T1, T2 <: Real }

    # fetch values
    z0, α, β = getall(f)
    
    # calculate result
    denominator = α + norm(input - z0) # Not sure whether this is the correct norm
    denominator /= β

    result = copy(input)
    result .-= z0
    result ./= denominator
    result .+= input

    # return result
    return result

end
forward(f::RadialMap{T1,T2}, input::T1) where { T1, T2 <: Real } = _forward(f, input)
Broadcast.broadcasted(::typeof(forward), f::RadialMap{T1,T2}, input::Array{T1,1}) where { T1, T2 <: Real } = broadcast(_forward, Ref(f), input)


# forward pass through the RadialMap function (univariate input)
function _forward(f::RadialMap{T1,T2}, input::T3) where { T1 <: Real, T2 <: Real, T3 <: Real }

    # fetch values
    z0, α, β = getall(f)
    
    # calculate result
    denominator = α + norm(input - z0) # Not sure whether this is the correct norm
    denominator /= β

    result = copy(input)
    result -= z0
    result /= denominator
    result += input

    # return result
    return result

end
forward(f::RadialMap{T1,T2}, input::T3) where { T1 <: Real, T2 <: Real, T3 <: Real } = _forward(f, input)
Broadcast.broadcasted(::typeof(forward), f::RadialMap{T1,T2}, input::Array{T3,1}) where { T1 <: Real, T2 <: Real, T3 <: Real } = broadcast(_forward, Ref(f), input)

# inplace forward pass through the RadialMap function (multivariate input)
function forward!(output::T1, f::RadialMap{T1,T2}, input::T1) where { T1, T2 <: Real }

    # check dimensionality
    @assert length(output) == length(input) "The length of the preallocated vector does not seem to match the length of the input vector."

    # fetch values
    z0, α, β = getall(f)
    
    # calculate result
    denominator = α + norm(input - z0) # Not sure whether this is the correct norm
    denominator /= β

    output .= input
    output .-= z0
    output ./= denominator
    output .+= input

end

# jacobian of the RadialMap function (multivariate input)
function _jacobian(f::RadialMap{T1,T2}, input::T1) where { T1, T2 <: Real}

    # fetch values 
    z0, α, β = getall(f)

    # # calculate result
    diff = input - z0
    result = diff*diff'
    hi = α + norm(diff)
    βh = β/hi
    ζ  = -β / hi / hi / norm(diff)
    result .*= ζ
    @inbounds for k = 1:length(input)
        result[k,k] += 1
        result[k,k] += βh
    end

    # return result
    return result

end
jacobian(f::RadialMap{T1,T2}, input::T1) where { T1, T2 <: Real } = _jacobian(f, input)
Broadcast.broadcasted(::typeof(jacobian), f::RadialMap{T1,T2}, input::Array{T1,1}) where { T1, T2 <: Real } = broadcast(_jacobian, Ref(f), input)

# jacobian of the RadialMap function (univariate input)
function _jacobian(f::RadialMap{T1,T2}, input::T3) where { T1 <: Real, T2 <: Real, T3 <: Real } 

    # fetch values 
    z0, α, β = getall(f)

    # calculate result (optimized)
    diff = input - z0
    h = 1 / (α + norm(diff))
    result = 1 + β*h - β * h * h /norm(diff) * diff * diff

    # return result
    return result

end
jacobian(f::RadialMap{T1,T2}, input::T3) where { T1 <: Real, T2 <: Real, T3 <: Real } = _jacobian(f, input)
Broadcast.broadcasted(::typeof(jacobian), f::RadialMap{T1,T2}, input::Array{T3,1}) where { T1 <: Real, T2 <: Real, T3 <: Real } = broadcast(_jacobian, Ref(f), input)

# inplace jacobian of the RadialMap function (multivariate input)
function jacobian!(output::Array{T2,2}, f::RadialMap{T1,T2}, input::T1) where { T1, T2 <: Real}

    # check whether the dimensionality is correct
    @assert size(output) == (length(input), length(f.z0)) "The dimensionality of the preallocated jacobian matrix seems incorrect."

    # fetch values 
    z0, α, β = getall(f)

    # # calculate result
    diff = input - z0
    for ku = 1:length(diff)
        for kw = 1:length(diff)
            output[ku,kw] = diff[ku]*diff[kw]
        end
    end
    hi = α + norm(diff)
    βh = β/hi
    ζ  = -β / hi / hi / norm(diff)
    output .*= ζ
    @inbounds for k = 1:length(input)
        output[k,k] += 1
        output[k,k] += βh
    end

end

# determinant of the jacobian of the RadialMap function (multivariate input)
det_jacobian(f::RadialMap{T1,T2}, input::T1) where { T1, T2 <: Real} = det(jacobian(f, input))

# determinant of the jacobian of the RadialMap function (univariate input)
function det_jacobian(f::RadialMap{T, T}, input::T) where { T <: Real }

    # fetch values
    z0, α, β = getall(f)

    # calculate result
    r = norm(input - z0)
    h = 1 / (α + r)
    result = ( 1 + β*h - β*h^2*r ) * ( 1 + β*h )^(length(input)-1)

    # return result
    return result

end

# extra utility function (multivariate)
inv_jacobian(f::RadialMap{T1,T2}, input::T1) where { T1, T2 <: Real }       = inv(jacobian(f, input))
absdet_jacobian(f::RadialMap{T1,T2}, input::T1) where { T1, T2 <: Real }    = abs(det_jacobian(f, input))
logabsdet_jacobian(f::RadialMap{T1,T2}, input::T1) where { T1, T2 <: Real}  = logabsdet(jacobian(f, input))

# extra utility functions (univariate)
inv_jacobian(f::RadialMap{T,T}, input::T) where { T <: Real }               = 1.0 / jacobian(f, input)
absdet_jacobian(f::RadialMap{T,T}, input::T) where { T <: Real }            = abs(det_jacobian(f, input))
logabsdet_jacobian(f::RadialMap{T,T}, input::T) where { T <: Real}          = log(absdet_jacobian(f, input))