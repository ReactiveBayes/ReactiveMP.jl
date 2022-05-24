export ExponentialLinearQuadratic

import Distributions: pdf, logpdf, ContinuousUnivariateDistribution
import Base: eltype
struct ExponentialLinearQuadratic{A <: AbstractApproximationMethod, T <: Real} <: ContinuousUnivariateDistribution
    approximation::A
    a::T
    b::T
    c::T
    d::T
end

Base.eltype(::Type{<:ExponentialLinearQuadratic{A, T}}) where {A, T} = T
Base.eltype(::ExponentialLinearQuadratic{A, T}) where {A, T}         = T

ExponentialLinearQuadratic(approximation, a::Real, b::Real, c::Real, d::Real)             = ExponentialLinearQuadratic(approximation, promote(a, b, c, d)...)
ExponentialLinearQuadratic(approximation, a::Integer, b::Integer, c::Integer, d::Integer) = ExponentialLinearQuadratic(approximation, float(a), float(b), float(c), float(d))

mean_cov(dist::ExponentialLinearQuadratic) = mean_var(dist)

function mean_var(dist::ExponentialLinearQuadratic)
    # This is equivalent to (x) -> pdf(dist, x) * exp(x^2 / 2)
    adjusted_pdf = let a = dist.a, b = dist.b, c = dist.c, d = dist.d
        (x) -> exp(-(a * x - x^2 + b * exp(c * x + d * x^2 / 2)) / 2)
    end
    return approximate_meancov(
        dist.approximation,
        adjusted_pdf,
        NormalMeanVariance(zero(eltype(dist)), one(eltype(dist)))
    )
end

mean_invcov(dist::ExponentialLinearQuadratic)      = mean_cov(dist) .|> (identity, inv)
mean_std(dist::ExponentialLinearQuadratic)         = mean_var(dist) .|> (identity, sqrt)
weightedmean_cov(dist::ExponentialLinearQuadratic) = weightedmean_var(dist)
weightedmean_std(dist::ExponentialLinearQuadratic) = weightedmean_var(dist) .|> (identity, sqrt)

function weightedmean_var(dist::ExponentialLinearQuadratic)
    m, v = mean_cov(dist)
    return (inv(v) * m, v)
end

function weightedmean_invcov(dist::ExponentialLinearQuadratic)
    m, w = mean_invcov(dist)
    return (w * m, w)
end

Distributions.pdf(dist::ExponentialLinearQuadratic, x::Real)    = exp(logpdf(dist, x))
Distributions.logpdf(dist::ExponentialLinearQuadratic, x::Real) = -(dist.a * x + dist.b * exp(dist.c * x + dist.d * x^2 / 2)) / 2
Distributions.mean(dist::ExponentialLinearQuadratic)            = mean_var(dist)[1]
Distributions.var(dist::ExponentialLinearQuadratic)             = mean_var(dist)[2]
Distributions.std(dist::ExponentialLinearQuadratic)             = mean_std(dist)[2]
Distributions.cov(dist::ExponentialLinearQuadratic)             = var(dist)

invcov(dist::ExponentialLinearQuadratic)       = mean_invcov(dist)[2]
precision(dist::ExponentialLinearQuadratic)    = mean_invcov(dist)[2]
weightedmean(dist::ExponentialLinearQuadratic) = weightedmean_invcov(dist)[1]

prod_analytical_rule(::Type{<:UnivariateNormalDistributionsFamily}, ::Type{<:ExponentialLinearQuadratic}) =
    ProdAnalyticalRuleAvailable()
prod_analytical_rule(::Type{<:ExponentialLinearQuadratic}, ::Type{<:UnivariateNormalDistributionsFamily}) =
    ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::UnivariateNormalDistributionsFamily, right::ExponentialLinearQuadratic)
    mean, variance = approximate_meancov(right.approximation, (z) -> pdf(right, z), left)
    return NormalMeanVariance(mean, variance)
end

function prod(::ProdAnalytical, left::ExponentialLinearQuadratic, right::UnivariateNormalDistributionsFamily)
    mean, variance = approximate_meancov(left.approximation, (z) -> pdf(left, z), right)
    return NormalMeanVariance(mean, variance)
end
