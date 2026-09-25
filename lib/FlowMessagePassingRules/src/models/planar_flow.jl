# The planar flow.

@doc raw"""
    PlanarFlow(u, w, b)
    PlanarFlow([rng,] dim::Int)
    PlanarFlow()

The PlanarFlow function is defined as

```math
f({\bf{x}}) = {\bf{x}} + {\bf{u}} \tanh({\bf{w}}^\top {\bf{x}} + b)
```

with input and output dimension ``D``. Here ``{\bf{x}}\in \mathbb{R}^D`` represents the input of the function. Furthermore ``{\bf{u}}\in \mathbb{R}^D``, ``{\bf{w}}\in \mathbb{R}^D`` and ``b\in\mathbb{R}`` represent the parameters of the function. The function contracts and expands the input space.

`PlanarFlow(u, w, b)` takes the parameters, `u` and `w` both scalars or both vectors of the same
length. `PlanarFlow([rng,] dim)` draws them from a standard (multivariate) normal distribution,
using `rng`, by default the task's generator: scalars for `dim == 1`, vectors of length `dim`
otherwise. `PlanarFlow()` is a placeholder whose dimension is set when it is wrapped in a model,
and whose parameters are set by [`compile`](@ref).

This function has been introduced in:

Rezende, Danilo, and Shakir Mohamed. "Variational inference with normalizing flows." _International conference on machine learning._ PMLR, 2015.
"""
mutable struct PlanarFlow{T1, T2 <: Real} <: AbstractCouplingFlow
    u::T1
    w::T1
    b::T2
    function PlanarFlow(u::T1, w::T1, b::T2) where {T1, T2 <: Real}
        @assert length(u) == length(w) "The parameters u and w in the PlanarFlow structure should have equal length."
        return new{T1, T2}(u, w, float(b))
    end
end
struct PlanarFlowEmpty{N} <: AbstractCouplingFlowEmpty end
PlanarFlowEmpty(dim::Int) = PlanarFlowEmpty{dim}()
struct PlanarFlowPlaceholder <: AbstractCouplingFlowPlaceholder end
PlanarFlow() = PlanarFlowPlaceholder()
function prepare(dim::Int, flow::PlanarFlowPlaceholder)
    return PlanarFlowEmpty(dim)
end

# The derivative of `tanh`, written here so as not to load BayesBase for it.
dtanh(x) = one(x) - abs2(tanh(x))

# compile placeholder, drawing the parameters from `rng`
compile(f::PlanarFlowEmpty) = compile(default_rng(), f)
compile(rng::AbstractRNG, f::PlanarFlowEmpty{1}) = PlanarFlow(randn(rng), randn(rng), randn(rng))
compile(rng::AbstractRNG, f::PlanarFlowEmpty) = PlanarFlow(randn(rng, getdim(f)), randn(rng, getdim(f)), randn(rng))
compile(f::PlanarFlowEmpty{1}, params) = PlanarFlow(params[1], params[2], params[3])
compile(f::PlanarFlowEmpty, params) = PlanarFlow(params[1:getdim(f)], params[(1 + getdim(f)):(2 * getdim(f))], params[2 * getdim(f) + 1])

PlanarFlow(dim::Int64) = PlanarFlow(default_rng(), dim)
function PlanarFlow(rng::AbstractRNG, dim::Int64)
    if dim == 1
        return PlanarFlow(randn(rng), randn(rng), randn(rng))
    else
        return PlanarFlow(randn(rng, dim), randn(rng, dim), randn(rng))
    end
end

# number of parameters
nr_params(flow::PlanarFlow) = length(flow.u) + length(flow.w) + length(flow.b)
nr_params(flow::PlanarFlowEmpty{N}) where {N} = 2 * N + 1

# get-functions for the PlanarFlow structure.
getu(f::PlanarFlow) = return f.u
getw(f::PlanarFlow) = return f.w
getb(f::PlanarFlow) = return f.b
getall(f::PlanarFlow) = return f.u, f.w, f.b
getdim(f::PlanarFlowEmpty{N}) where {N} = return N

# set-functions for the PlanarFlow structure
function setu!(f::PlanarFlow{T1, T2}, u::T1) where {T1, T2 <: Real}
    @assert length(f.u) == length(u) "The dimensionality of the current value of u and its new value do not match."
    return f.u = u
end

function setw!(f::PlanarFlow{T1, T2}, w::T1) where {T1, T2 <: Real}
    @assert length(f.w) == length(w) "The dimensionality of the current value of w and its new value do not match."
    return f.w = w
end

function setb!(f::PlanarFlow{T1, T2}, b::T2) where {T1, T2 <: Real}
    return f.b = b
end

# custom Base function for the PlanarFlow structure
Base.eltype(f::PlanarFlow{T1, T2}) where {T1 <: Real, T2 <: Real} = promote_type(T1, T2)
Base.eltype(f::PlanarFlow{T1, T2}) where {T1 <: AbstractVector, T2 <: Real} = promote_type(eltype(T1), T2)
Base.eltype(::Type{PlanarFlow{T1, T2}}) where {T1 <: Real, T2 <: Real} = promote_type(T1, T2)
Base.eltype(::Type{PlanarFlow{T1, T2}}) where {T1 <: AbstractVector, T2 <: Real} = promote_type(eltype(T1), T2)

Base.size(f::PlanarFlow{T1, T2}) where {T1 <: Real, T2 <: Real} = 1
Base.size(f::PlanarFlow{T1, T2}) where {T1 <: AbstractVector, T2 <: Real} = length(f.u)
Base.size(f::PlanarFlowEmpty{N}) where {N} = return N

Base.length(f::PlanarFlow{T1, T2}) where {T1 <: Real, T2 <: Real} = 1
Base.length(f::PlanarFlow{T1, T2}) where {T1 <: AbstractVector, T2 <: Real} = length(f.u)
Base.length(f::PlanarFlowEmpty{N}) where {N} = return N

# forward pass through the PlanarFlow function (multivariate input)
function _forward(f::PlanarFlow{T1, T2}, input::T1) where {T1, T2 <: Real}

    u, w, b = getall(f)

    result = u .* ones(size(u))
    result .*= tanh(dot(w, input) + b)
    result .+= input

    return result
end
forward(f::PlanarFlow{T1, T2}, input::T1) where {T1, T2 <: Real} =
    _forward(f, input)
Broadcast.broadcasted(
    ::typeof(forward), f::PlanarFlow{T1, T2}, input::AbstractVector{T1}
) where {T1, T2 <: Real} = broadcast(_forward, Ref(f), input)

# forward pass through the PlanarFlow function (univariate input)
function _forward(
        f::PlanarFlow{T1, T2}, input::T3
    ) where {T1 <: Real, T2 <: Real, T3 <: Real}

    u, w, b = getall(f)

    result = copy(u)
    result *= tanh(dot(w, input) + b)
    result += input

    return result
end
forward(
    f::PlanarFlow{T1, T2}, input::T3
) where {T1 <: Real, T2 <: Real, T3 <: Real} = _forward(f, input)
Broadcast.broadcasted(
    ::typeof(forward), f::PlanarFlow{T1, T2}, input::AbstractVector{<:Real}
) where {T1 <: Real, T2 <: Real} = broadcast(_forward, Ref(f), input)

function _forward(f::PlanarFlow{T1, T2}, input) where {T1 <: Real, T2 <: Real}
    # function when the input is an array with 1 element
    @assert length(input) == 1 "Something is wrong with the dimensionality of the input to the PlanarFlow flow."
    return forward(f, input[1])
end
forward(f::PlanarFlow{T1, T2}, input) where {T1 <: Real, T2 <: Real} =
    _forward(f, input)
Broadcast.broadcasted(
    ::typeof(forward), f::PlanarFlow{T1, T2}, input::AbstractVector
) where {T1 <: Real, T2 <: Real} = broadcast(_forward, Ref(f), input)

# inplace forward pass through the PlanarFlow function (multivariate input)
function forward!(
        output::T1, f::PlanarFlow{T1, T2}, input::T1
    ) where {T1, T2 <: Real}

    @assert length(output) == length(input) "The length of the preallocated vector does not seem to match the length of the input vector."

    u, w, b = getall(f)

    output .= u
    output .*= tanh(dot(w, input) + b)
    return output .+= input
end

# jacobian of the PlanarFlow function (multivariate input)
function _jacobian(f::PlanarFlow{T1, T2}, input::T1) where {T1, T2 <: Real}

    u, w, b = getall(f)

    result = u * w'
    result .*= dtanh(dot(w, input) + b)
    @inbounds for k in 1:length(input)
        result[k, k] += 1.0
    end

    return result
end
jacobian(f::PlanarFlow{T1, T2}, input::T1) where {T1, T2 <: Real} =
    _jacobian(f, input)
Broadcast.broadcasted(
    ::typeof(jacobian), f::PlanarFlow{T1, T2}, input::AbstractVector{T1}
) where {T1, T2 <: Real} = broadcast(_jacobian, Ref(f), input)

# jacobian of the PlanarFlow function (univariate input)
function _jacobian(
        f::PlanarFlow{T1, T2}, input::T3
    ) where {T1 <: Real, T2 <: Real, T3 <: Real}

    u, w, b = getall(f)

    # calculate result (optimized)
    result = u * w * dtanh(w * input + b) + 1

    return result
end
jacobian(
    f::PlanarFlow{T1, T2}, input::T3
) where {T1 <: Real, T2 <: Real, T3 <: Real} = _jacobian(f, input)
Broadcast.broadcasted(
    ::typeof(jacobian), f::PlanarFlow{T1, T2}, input::AbstractVector{<:Real}
) where {T1 <: Real, T2 <: Real} = broadcast(_jacobian, Ref(f), input)

function _jacobian(
        f::PlanarFlow{T1, T2}, input::T3
    ) where {T1 <: Real, T2 <: Real, T3}
    # function when the input is an array with 1 element
    @assert length(input) == 1 "Something is wrong with the dimensionality of the input to the PlanarFlow flow."
    return jacobian(f, input[1])
end
jacobian(f::PlanarFlow{T1, T2}, input::T3) where {T1 <: Real, T2 <: Real, T3} =
    _jacobian(f, input)
Broadcast.broadcasted(
    ::typeof(jacobian), f::PlanarFlow{T1, T2}, input::AbstractVector
) where {T1 <: Real, T2 <: Real} = broadcast(_jacobian, Ref(f), input)

# inplace jacobian of the PlanarFlow function (multivariate input)
function jacobian!(
        output::AbstractMatrix{T2}, f::PlanarFlow{T1, T2}, input::T1
    ) where {T1, T2 <: Real}

    @assert size(output) == (length(input), length(f.u)) "The dimensionality of the preallocated jacobian matrix seems incorrect."

    u, w, b = getall(f)

    for ku in 1:length(u)
        for kw in 1:length(w)
            output[ku, kw] = u[ku] * w[kw]
        end
    end
    output .*= dtanh(dot(w, input) + b)
    return @inbounds for k in 1:length(input)
        output[k, k] += 1.0
    end
end

# determinant of the jacobian of the PlanarFlow function (multivariate input)
det_jacobian(f::PlanarFlow{T1, T2}, input::T1) where {T1, T2 <: Real} =
    det(jacobian(f, input))

# determinant of the jacobian of the PlanarFlow function (univariate input)
function det_jacobian(f::PlanarFlow{T, T}, input::T) where {T <: Real}

    u, w, b = getall(f)

    result = 1 + dot(u, w) * dtanh(dot(w, input) + b)

    return result
end

# extra utility function (multivariate)
inv_jacobian(f::PlanarFlow{T1, T2}, input::T1) where {T1, T2 <: Real} = inv(jacobian(f, input))
absdet_jacobian(f::PlanarFlow{T1, T2}, input::T1) where {T1, T2 <: Real} = abs(det_jacobian(f, input))
logabsdet_jacobian(f::PlanarFlow{T1, T2}, input::T1) where {T1, T2 <: Real} = logabsdet(jacobian(f, input))

# extra utility functions (univariate)
inv_jacobian(f::PlanarFlow{T, T}, input::T) where {T <: Real} = 1.0 / jacobian(f, input)
absdet_jacobian(f::PlanarFlow{T, T}, input::T) where {T <: Real} = abs(det_jacobian(f, input))
logabsdet_jacobian(f::PlanarFlow{T, T}, input::T) where {T <: Real} = log(absdet_jacobian(f, input))

function inv_jacobian(
        f::PlanarFlow{T1, T2}, input
    ) where {T1 <: Real, T2 <: Real}
    # function when the input is an array with 1 element
    @assert length(input) == 1 "Something is wrong with the dimensionality of the input to the PlanarFlow flow."
    return inv_jacobian(f, input[1])
end
