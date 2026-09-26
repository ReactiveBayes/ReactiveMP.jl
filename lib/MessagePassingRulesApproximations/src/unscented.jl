const default_alpha = 1.0e-3 # Default value for the spread parameter
const default_beta = 2.0
const default_kappa = 0.0

struct UnscentedExtra{T, R, M, C}
    L::T
    λ::R
    Wm::M
    Wc::C
end

"""
    Unscented(; alpha = 1e-3, beta = 2.0, kappa = 0.0)
    Unscented(dim::Int; alpha = 1e-3, beta = 2.0, kappa = 0.0)

The unscented transform: the mean and covariance of `f(x)` for a normal `x`, estimated from
`2d + 1` deterministically placed sigma points, `d` being the dimension of `x`. It needs no
derivatives and captures the mean and covariance of `f(x)` to second order. [`UT`](@ref) and
[`UnscentedTransform`](@ref) are aliases.

# Keywords

- `alpha`: the spread of the sigma points around the mean, a small positive number. Default
  `1e-3`;
- `beta`: prior knowledge of the input's distribution, `2` being optimal for a normal. Default
  `2.0`;
- `kappa`: the secondary scaling. Default `0.0`.

With `dim`, the weights for that dimension are computed once and stored with the method;
without it, they are computed on each call, for whatever dimension the inputs have.

It is used through [`approximate`](@ref), for the output's moments, and
[`unscented_statistics`](@ref), which adds the cross-covariance [`smoothRTS`](@ref) needs.
[`Linearization`](@ref) is the alternative that expands the function instead.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> m, V = approximate(Unscented(), sin, (0.0,), (1e-4,));

julia> isapprox(m, 0.0; atol = 1e-12) && isapprox(V, 1e-4; rtol = 1e-3)
true
```
"""
struct Unscented{A, B, K, E} <: AbstractApproximationMethod
    α::A
    β::B
    κ::K
    e::E
end

# Structure constructor
function Unscented(;
        alpha::A = default_alpha, beta::B = default_beta, kappa::K = default_kappa
    ) where {A <: Real, B <: Real, K <: Real}
    return Unscented{A, B, K, Nothing}(alpha, beta, kappa, nothing)
end

function Unscented(
        dim::Int64;
        alpha::Real = default_alpha,
        beta::Real = default_beta,
        kappa::Real = default_kappa,
    )
    α = alpha
    β = beta
    κ = kappa
    λ = α^2 * (dim + κ) - dim
    Wm = ones(2 * dim + 1)
    Wc = ones(2 * dim + 1)
    Wm ./= (2 * (dim + λ))
    Wc ./= (2 * (dim + λ))
    Wm[1] = λ / (dim + λ)
    Wc[1] = λ / (dim + λ) + (1 - α^2 + β)
    return Unscented(α, β, κ, UnscentedExtra(dim, λ, Wm, Wc))
end

"""
    UT

An alias for [`Unscented`](@ref), the unscented transform.
"""
const UT = Unscented

"""
    UnscentedTransform

An alias for [`Unscented`](@ref), the unscented transform.
"""
const UnscentedTransform = Unscented

approximation_name(::Unscented) = "Unscented"
approximation_short_name(::Unscented) = "UT"

# get-functions for the Unscented structure

getα(approximation::Unscented) = approximation.α
getβ(approximation::Unscented) = approximation.β
getκ(approximation::Unscented) = approximation.κ

getextra(approximation::Unscented) = approximation.e

getL(approximation::Unscented) = getL(getextra(approximation))
getλ(approximation::Unscented) = getλ(getextra(approximation))
getWm(approximation::Unscented) = getWm(getextra(approximation))
getWc(approximation::Unscented) = getWc(getextra(approximation))

getL(extra::UnscentedExtra) = extra.L
getλ(extra::UnscentedExtra) = extra.λ
getWm(extra::UnscentedExtra) = extra.Wm
getWc(extra::UnscentedExtra) = extra.Wc

# Copied and refactored from ForneyLab.jl

"""
    approximate(method::Unscented, f, means::Tuple, covs::Tuple) -> (m, V)

The mean and covariance of `f(x₁, x₂, …)` for independent normal inputs, by the unscented
transform.

# Arguments

- `f`: a function of one argument per input, returning a number or a vector;
- `means`, `covs`: one entry per input, a number and a variance, or a vector and a covariance
  matrix. Several inputs are treated jointly, as one normal whose covariance is block diagonal.

# Returns

`(m, V)`: the output's mean and variance, or mean vector and covariance matrix.

# Throws

A `DomainError` when a covariance is infinite. A zero covariance is not an error: the input is a
point, and the result is `f` at it with zero covariance, after a warning.
"""
function approximate(
        method::Unscented, f::F, means::Tuple, covs::Tuple
    ) where {F}
    # `Val(false)` indicates that we do not compute the `C` component
    (m, V, _) = unscented_statistics(method, Val(false), f, means, covs)
    return (m, V)
end

"""
    unscented_statistics(method::Unscented, g, means::Tuple, covs::Tuple) -> (m, V, C)

The output's mean `m` and covariance `V`, as [`approximate`](@ref)`(::Unscented, …)` computes
them, and the cross-covariance `C` between the inputs, concatenated into one vector, and the
output: the forward statistics [`smoothRTS`](@ref) takes. Arguments and errors are those of
`approximate`; for a point input, `C` is zero.
"""
function unscented_statistics(
        method::Unscented, g::G, means::Tuple, covs::Tuple
    ) where {G}
    # By default we compute the `C` component, thus `Val(true)`
    return unscented_statistics(method, Val(true), g, means, covs)
end

function statistic_estimation(
        ::Val{C}, first_element::T, g_sigma, sigma_points, m, weights_m, weights_c
    ) where {C, T <: Real}
    m_tilde = sum(weights_m .* g_sigma)
    V_tilde = sum(weights_c .* (g_sigma .- m_tilde) .^ 2)

    # Compute `C_tilde` only if `C === true`
    C_tilde = if C
        sum(weights_c .* (sigma_points .- m) .* (g_sigma .- m_tilde))
    else
        nothing
    end
    return (m_tilde, V_tilde, C_tilde)
end

function statistic_estimation(
        ::Val{C}, first_element::V, g_sigma, sigma_points, m, weights_m, weights_c
    ) where {C, V <: AbstractVector}
    d_out = length(first(g_sigma))

    @inbounds m_tilde = sum(wm * yi for (wm, yi) in zip(weights_m, g_sigma))
    @inbounds V_tilde = sum(
        wc * (yi - m_tilde) * (yi - m_tilde)' for
            (wc, yi) in zip(weights_c, g_sigma)
    )

    # Compute `C_tilde` only if `C === true`
    @inbounds C_tilde = if C
        reshape(
            sum(
                wc * (xi - m) * (yi - m_tilde) for
                    (wc, xi, yi) in zip(weights_c, sigma_points, g_sigma)
            ),
            1,
            d_out,
        )
    else
        nothing
    end
    return (m_tilde, V_tilde, C_tilde)
end

# A zero-covariance input means the input is known exactly, so the transformed output is a
# point too: zero output covariance *and* zero cross-covariance with the input.
#
# The cross-covariance is a genuine zero, not `nothing` ("not computed"): consumers that ask
# for it (`Val(true)`, i.e. the `DeltaFn(:ins)` marginal rules) do arithmetic with it, and
# callers that do not ask discard the third element anyway, so a zero is correct for both and
# keeps the return type stable.
__unscented_parameters_zero_covariance(m::T) where {T <: Real} =
    (m, zero(T), zero(T))
__unscented_parameters_zero_covariance(m::AbstractVector{T}) where {T <: Real} =
    (m, zeros(T, length(m), length(m)), zeros(T, length(m), length(m)))

# Single univariate variable
function unscented_statistics(
        method::Unscented, ::Val{C}, g::G, means::Tuple{Real}, covs::Tuple{Real}
    ) where {C, G}
    m = first(means)
    V = first(covs)
    if V == 0.0
        @warn "Unscented transform called with zero covariance input (function $g)" maxlog = 1
        resulting_m = g(m)
        return __unscented_parameters_zero_covariance(resulting_m)
    end

    if any(isinf, V)
        throw(
            DomainError(
                "unscented_statistics cannot be computed with infinite variance $covs",
            ),
        )
    end

    (sigma_points, weights_m, weights_c) = sigma_points_weights(method, m, V)

    # Evaluate g at sigma points
    g_sigma = g.(sigma_points)

    # Compute output statistics depending on the output variate type
    return statistic_estimation(
        Val(C), first(g_sigma), g_sigma, sigma_points, m, weights_m, weights_c
    )
end

# Single multivariate inbound
function unscented_statistics(
        method::Unscented,
        ::Val{C},
        g::G,
        means::Tuple{AbstractVector},
        covs::Tuple{AbstractMatrix},
    ) where {C, G}
    m = first(means)
    V = first(covs)
    if any(isinf, V)
        throw(
            DomainError(
                "unscented_statistics cannot be computed with infinite variance $covs",
            ),
        )
    end

    if all(vec -> all(x -> x == 0.0, vec), covs)
        @warn "Unscented transform called with zero covariance input (function $g)"
        resulting_m = g(m)
        return __unscented_parameters_zero_covariance(resulting_m)
    end

    (sigma_points, weights_m, weights_c) = sigma_points_weights(method, m, V)

    d = length(m)
    g_sigma = g.(sigma_points)
    @inbounds m_tilde = sum(weights_m[k + 1] * g_sigma[k + 1] for k in 0:(2d))
    @inbounds V_tilde = sum(
        weights_c[k + 1] *
            ((g_sigma[k + 1] - m_tilde) * (g_sigma[k + 1] - m_tilde)') for
            k in 0:(2d)
    )

    # Compute `C_tilde` only if `C === true`
    @inbounds C_tilde = if C
        sum(
            weights_c[k + 1] *
                (sigma_points[k + 1] - m) *
                (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)
        )
    else
        nothing
    end
    return (m_tilde, V_tilde, C_tilde)
end

# Multiple inbounds of possibly mixed variate type
function unscented_statistics(
        method::Unscented, ::Val{C}, g::G, ms::Tuple, Vs::Tuple
    ) where {C, G}
    (m, V, ds) = joint_mean_cov(ms, Vs)

    (sigma_points, weights_m, weights_c) = sigma_points_weights(method, m, V)

    g_sigma = [g(__splitjoin(sp, ds)...) for sp in sigma_points] # Unpack each sigma point in g

    d = sum(prod.(ds)) # Dimensionality of joint
    @inbounds m_tilde = sum(weights_m[k + 1] * g_sigma[k + 1] for k in 0:(2d)) # Vector
    @inbounds V_tilde = sum(
        weights_c[k + 1] *
            ((g_sigma[k + 1] - m_tilde) * (g_sigma[k + 1] - m_tilde)') for
            k in 0:(2d)
    ) # Matrix

    # Compute `C_tilde` only if `C === true`
    @inbounds C_tilde = if C
        sum(
            weights_c[k + 1] *
                (sigma_points[k + 1] - m) *
                (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)
        )
    else
        nothing
    end

    return (m_tilde, V_tilde, C_tilde)
end

"""
    sigma_points_weights(method::Unscented, m::Real, V::Real) -> (points, weights_m, weights_c)
    sigma_points_weights(method::Unscented, m::AbstractVector, V::AbstractMatrix) -> (points, weights_m, weights_c)

The `2d + 1` sigma points of `N(m, V)` under `method`'s parameters, and their weights for the mean
and for the covariance, in the same order: tuples for a scalar normal, vectors otherwise. The
first point is the mean. It warns when the parameters make `d + λ` negative, which gives
unreliable estimates.
"""
function sigma_points_weights(method::Unscented, m::Real, V::Real)
    alpha = getα(method)
    beta = getβ(method)
    kappa = getκ(method)
    lambda = (1 + kappa) * alpha^2 - 1

    if (1 + lambda) < 0
        @warn "`(1 + lambda)` in the sigma points computation routine is negative. This may lead to the incorrect results. Adjust the `alpha`, `kappa` and `beta` parameters."
    end

    l = sqrt((1 + lambda) * V)

    sigma_points = (m, m + l, m - l)
    weights_m = (lambda / (1 + lambda), 1 / (2 * (1 + lambda)), 1 / (2 * (1 + lambda)))
    weights_c = (weights_m[1] + (1 - alpha^2 + beta), 1 / (2 * (1 + lambda)), 1 / (2 * (1 + lambda)))

    return (sigma_points, weights_m, weights_c)
end

function sigma_points_weights(
        method::Unscented, m::AbstractVector, V::AbstractMatrix
    )
    d = length(m)
    alpha = getα(method)
    beta = getβ(method)
    kappa = getκ(method)
    lambda = (d + kappa) * alpha^2 - d

    if (d + lambda) < 0
        @warn "`(d + lambda)` in the sigma points computation routine is negative. This may lead to the incorrect results. Adjust the `alpha`, `kappa` and `beta` parameters."
    end

    T = promote_type(eltype(m), eltype(V))

    sigma_points = Vector{Vector{T}}(undef, 2 * d + 1)
    weights_m = Vector{T}(undef, 2 * d + 1)
    weights_c = Vector{T}(undef, 2 * d + 1)

    L = cholsqrt((d + lambda) * V)

    sigma_points[1] = m
    weights_m[1] = lambda / (d + lambda)
    weights_c[1] = weights_m[1] + (1 - alpha^2 + beta)

    @inbounds for i in 1:d
        @views sigma_points[2 * i] = m + L[:, i]
        @views sigma_points[2 * i + 1] = m - L[:, i]
    end

    @inbounds weights_m[2:end] .= 1 / (2 * (d + lambda))
    @inbounds weights_c[2:end] .= 1 / (2 * (d + lambda))

    return (sigma_points, weights_m, weights_c)
end
