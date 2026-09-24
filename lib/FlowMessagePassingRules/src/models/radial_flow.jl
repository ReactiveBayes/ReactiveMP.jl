# The radial flow, from v6's `coupling_flows/radial_flow.jl`.

@doc raw"""
    RadialFlow(z0, α, β)
    RadialFlow([rng,] dim::Int)
    RadialFlow()

The RadialFlow function is defined as

```math
f({\bf{x}}) = {\bf{x}} + \frac{\beta({\bf{x}} - {\bf{z}}_0)}{\alpha + |{\bf{x}} - {\bf{z}}_0|}
```

with input and output dimension ``D``. Here ``{\bf{x}}\in \mathbb{R}^D`` represents the input of the function. Furthermore ``{\bf{z}}_0\in \mathbb{R}^D``, ``\alpha\in \mathbb{R}`` and ``\beta\in\mathbb{R}`` represent the parameters of the function. The function contracts and expands the input space.

`RadialFlow(z0, α, β)` takes the parameters, with `α > 0`. `RadialFlow([rng,] dim)` draws them
using `rng`, by default the task's generator: `z0` and `β` from a standard (multivariate) normal
distribution, `α` uniformly from ``[0, 1)``. `RadialFlow()` is a placeholder whose dimension is set
when it is wrapped in a model, and whose parameters are set by [`compile`](@ref).

This function has been introduced in:

Rezende, Danilo, and Shakir Mohamed. "Variational inference with normalizing flows." _International conference on machine learning._ PMLR, 2015.
"""
mutable struct RadialFlow{T1, T2 <: Real} <: AbstractCouplingFlow
    z0::T1
    α::T2
    β::T2
    function RadialFlow(z0::T1, α::T2, β::T2) where {T1, T2 <: Real}
        @assert α > 0 "The parameter α should be larger than 0."
        return new{T1, T2}(z0, float(α), float(β))
    end
end
struct RadialFlowEmpty{N} <: AbstractCouplingFlowEmpty end
RadialFlowEmpty(dim::Int) = RadialFlowEmpty{dim}()
struct RadialFlowPlaceholder <: AbstractCouplingFlowPlaceholder end
RadialFlow() = RadialFlowPlaceholder()
function prepare(dim::Int, flow::RadialFlowPlaceholder)
    return RadialFlowEmpty(dim)
end

# compile placeholder, drawing the parameters from `rng`
compile(f::RadialFlowEmpty) = compile(default_rng(), f)
compile(rng::AbstractRNG, f::RadialFlowEmpty{1}) = RadialFlow(randn(rng), rand(rng), randn(rng))
compile(rng::AbstractRNG, f::RadialFlowEmpty) = RadialFlow(randn(rng, getdim(f)), rand(rng), randn(rng))
compile(f::RadialFlowEmpty{1}, params) = RadialFlow(params[1], params[2], params[3])
compile(f::RadialFlowEmpty, params) = RadialFlow(params[1:getdim(f)], params[getdim(f) + 1], params[getdim(f) + 2])

RadialFlow(dim::Int64) = RadialFlow(default_rng(), dim)
RadialFlow(rng::AbstractRNG, dim::Int64) = RadialFlow(randn(rng, dim), rand(rng), randn(rng))

# number of parameters
nr_params(flow::RadialFlow) = 2 + length(flow.z0)
nr_params(flow::RadialFlowEmpty) = 2 + getdim(flow)

# get-functions for the RadialFlow structure.
getz0(f::RadialFlow) = return f.z0
getα(f::RadialFlow) = return f.α
getβ(f::RadialFlow) = return f.β
getall(f::RadialFlow) = return f.z0, f.α, f.β
getdim(f::RadialFlowEmpty{N}) where {N} = return N

# set-functions for the RadialFlow structure
function setz0!(f::RadialFlow{T1, T2}, z0::T1) where {T1, T2 <: Real}
    @assert length(f.z0) == length(z0) "The dimensionality of the current value of z0 and its new value do not match."
    return f.z0 = z0
end

function setα!(f::RadialFlow{T1, T2}, α::T2) where {T1, T2 <: Real}
    return f.α = α
end

function setβ!(f::RadialFlow{T1, T2}, β::T2) where {T1, T2 <: Real}
    return f.β = β
end

# custom Base function for the RadialFlow structure
Base.eltype(f::RadialFlow{T1, T2}) where {T1 <: Real, T2 <: Real} = promote_type(T1, T2)
Base.eltype(f::RadialFlow{T1, T2}) where {T1 <: AbstractVector, T2 <: Real} = promote_type(eltype(T1), T2)
Base.eltype(::Type{RadialFlow{T1, T2}}) where {T1 <: Real, T2 <: Real} = promote_type(T1, T2)
Base.eltype(::Type{RadialFlow{T1, T2}}) where {T1 <: AbstractVector, T2 <: Real} = promote_type(eltype(T1), T2)

Base.size(f::RadialFlow{T1, T2}) where {T1 <: Real, T2 <: Real} = 1
Base.size(f::RadialFlow{T1, T2}) where {T1 <: AbstractVector, T2 <: Real} = length(f.z0)
Base.size(f::RadialFlowEmpty{N}) where {N} = return N

Base.length(f::RadialFlow{T1, T2}) where {T1 <: Real, T2 <: Real} = 1
Base.length(f::RadialFlow{T1, T2}) where {T1 <: AbstractVector, T2 <: Real} = length(f.z0)
Base.length(f::RadialFlowEmpty{N}) where {N} = return N

# forward pass through the RadialFlow function (multivariate input)
function _forward(f::RadialFlow{T1, T2}, input::T1) where {T1, T2 <: Real}

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
forward(f::RadialFlow{T1, T2}, input::T1) where {T1, T2 <: Real} =
    _forward(f, input)
Broadcast.broadcasted(
    ::typeof(forward), f::RadialFlow{T1, T2}, input::AbstractVector{T1}
) where {T1, T2 <: Real} = broadcast(_forward, Ref(f), input)

# forward pass through the RadialFlow function (univariate input)
function _forward(
        f::RadialFlow{T1, T2}, input::T3
    ) where {T1 <: Real, T2 <: Real, T3 <: Real}

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
forward(
    f::RadialFlow{T1, T2}, input::T3
) where {T1 <: Real, T2 <: Real, T3 <: Real} = _forward(f, input)
Broadcast.broadcasted(
    ::typeof(forward), f::RadialFlow{T1, T2}, input::AbstractVector{<:Real}
) where {T1 <: Real, T2 <: Real} = broadcast(_forward, Ref(f), input)

function _forward(f::RadialFlow{T1, T2}, input) where {T1 <: Real, T2 <: Real}
    # function when the input is an array with 1 element
    @assert length(input) == 1 "Something is wrong with the dimensionality of the input to the RadialFlow flow."
    return forward(f, input[1])
end
forward(f::RadialFlow{T1, T2}, input) where {T1 <: Real, T2 <: Real} =
    _forward(f, input)
Broadcast.broadcasted(
    ::typeof(forward), f::RadialFlow{T1, T2}, input::AbstractVector
) where {T1 <: Real, T2 <: Real} = broadcast(_forward, Ref(f), input)

# inplace forward pass through the RadialFlow function (multivariate input)
function forward!(
        output::T1, f::RadialFlow{T1, T2}, input::T1
    ) where {T1, T2 <: Real}

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
    return output .+= input
end

# jacobian of the RadialFlow function (multivariate input)
function _jacobian(f::RadialFlow{T1, T2}, input::T1) where {T1, T2 <: Real}

    # fetch values
    z0, α, β = getall(f)

    # # calculate result
    diff = input - z0
    result = diff * diff'
    hi = α + norm(diff)
    βh = β / hi
    ζ = -β / hi / hi / norm(diff)
    result .*= ζ
    @inbounds for k in 1:length(input)
        result[k, k] += 1
        result[k, k] += βh
    end

    # return result
    return result
end
jacobian(f::RadialFlow{T1, T2}, input::T1) where {T1, T2 <: Real} =
    _jacobian(f, input)
Broadcast.broadcasted(
    ::typeof(jacobian), f::RadialFlow{T1, T2}, input::AbstractVector{T1}
) where {T1, T2 <: Real} = broadcast(_jacobian, Ref(f), input)

function _jacobian(f::RadialFlow{T1, T2}, input) where {T1 <: Real, T2 <: Real}
    # function when the input is an array with 1 element
    @assert length(input) == 1 "Something is wrong with the dimensionality of the input to the RadialFlow flow."
    return jacobian(f, input[1])
end
jacobian(f::RadialFlow{T1, T2}, input) where {T1 <: Real, T2 <: Real} =
    _jacobian(f, input)
Broadcast.broadcasted(
    ::typeof(jacobian), f::RadialFlow{T1, T2}, input::AbstractVector
) where {T1 <: Real, T2 <: Real} = broadcast(_jacobian, Ref(f), input)

# jacobian of the RadialFlow function (univariate input)
function _jacobian(
        f::RadialFlow{T1, T2}, input::T3
    ) where {T1 <: Real, T2 <: Real, T3 <: Real}

    # fetch values
    z0, α, β = getall(f)

    # calculate result (optimized)
    diff = input - z0
    h = 1 / (α + norm(diff))
    result = 1 + β * h - β * h * h / norm(diff) * diff * diff

    # return result
    return result
end
jacobian(
    f::RadialFlow{T1, T2}, input::T3
) where {T1 <: Real, T2 <: Real, T3 <: Real} = _jacobian(f, input)
Broadcast.broadcasted(
    ::typeof(jacobian), f::RadialFlow{T1, T2}, input::AbstractVector{<:Real}
) where {T1 <: Real, T2 <: Real} = broadcast(_jacobian, Ref(f), input)

# inplace jacobian of the RadialFlow function (multivariate input)
function jacobian!(
        output::AbstractMatrix{T2}, f::RadialFlow{T1, T2}, input::T1
    ) where {T1, T2 <: Real}

    # check whether the dimensionality is correct
    @assert size(output) == (length(input), length(f.z0)) "The dimensionality of the preallocated jacobian matrix seems incorrect."

    # fetch values
    z0, α, β = getall(f)

    # # calculate result
    diff = input - z0
    for ku in 1:length(diff)
        for kw in 1:length(diff)
            output[ku, kw] = diff[ku] * diff[kw]
        end
    end
    hi = α + norm(diff)
    βh = β / hi
    ζ = -β / hi / hi / norm(diff)
    output .*= ζ
    return @inbounds for k in 1:length(input)
        output[k, k] += 1
        output[k, k] += βh
    end
end

# determinant of the jacobian of the RadialFlow function (multivariate input)
det_jacobian(f::RadialFlow{T1, T2}, input::T1) where {T1, T2 <: Real} =
    det(jacobian(f, input))

# determinant of the jacobian of the RadialFlow function (univariate input)
function det_jacobian(f::RadialFlow{T, T}, input::T) where {T <: Real}

    # fetch values
    z0, α, β = getall(f)

    # calculate result
    r = norm(input - z0)
    h = 1 / (α + r)
    result = (1 + β * h - β * h^2 * r) * (1 + β * h)^(length(input) - 1)

    # return result
    return result
end

# extra utility function (multivariate)
inv_jacobian(f::RadialFlow{T1, T2}, input::T1) where {T1, T2 <: Real} = inv(jacobian(f, input))
absdet_jacobian(f::RadialFlow{T1, T2}, input::T1) where {T1, T2 <: Real} = abs(det_jacobian(f, input))
logabsdet_jacobian(f::RadialFlow{T1, T2}, input::T1) where {T1, T2 <: Real} = logabsdet(jacobian(f, input))

# extra utility functions (univariate)
inv_jacobian(f::RadialFlow{T, T}, input::T) where {T <: Real} = 1.0 / jacobian(f, input)
absdet_jacobian(f::RadialFlow{T, T}, input::T) where {T <: Real} = abs(det_jacobian(f, input))
logabsdet_jacobian(f::RadialFlow{T, T}, input::T) where {T <: Real} = log(absdet_jacobian(f, input))

function inv_jacobian(
        f::RadialFlow{T1, T2}, input::T3
    ) where {T1 <: Real, T2 <: Real, T3}
    # function when the input is an array with 1 element
    @assert length(input) == 1 "Something is wrong with the dimensionality of the input to the RadialFlow flow."
    return inv_jacobian(f, input[1])
end
