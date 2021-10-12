export ExponentialLinearQuadratic

import Distributions: pdf, logpdf, ContinuousUnivariateDistribution

struct ExponentialLinearQuadratic{ A <: AbstractApproximationMethod, T <: Real } <: ContinuousUnivariateDistribution
    approximation :: A
    a :: T
    b :: T
    c :: T
    d :: T
end

ExponentialLinearQuadratic(approximation, a::Real, b::Real, c::Real, d::Real)             = ExponentialLinearQuadratic(approximation, promote(a, b, c, d)...)
ExponentialLinearQuadratic(approximation, a::Integer, b::Integer, c::Integer, d::Integer) = ExponentialLinearQuadratic(approximation, float(a), float(b), float(c), float(d))

mean_cov(dist::ExponentialLinearQuadratic) = mean_var(dist)
mean_var(dist::ExponentialLinearQuadratic) = approximate_meancov(dist.approximation, (z) -> pdf(dist, z) * exp(0.5 * z^2), NormalMeanVariance(0.0, 1.0))

function mean_invcov(dist::ExponentialLinearQuadratic) 
    m, v = mean_cov(dist)
    return (m, inv(v))
end

function weightedmean_invcov(dist::ExponentialLinearQuadratic)
    m, w = mean_invcov(dist)
    return (w * m, w)
end

Distributions.pdf(dist::ExponentialLinearQuadratic, x::Real)    = exp(logpdf(dist, x))
Distributions.logpdf(dist::ExponentialLinearQuadratic, x::Real) = -0.5 * (dist.a * x + dist.b * exp(dist.c * x + dist.d * x ^ 2 / 2.0))
Distributions.mean(dist::ExponentialLinearQuadratic)            = mean_var(dist)[1]
Distributions.var(dist::ExponentialLinearQuadratic)             = mean_var(dist)[2]
Distributions.cov(dist::ExponentialLinearQuadratic)             = var(dist)

precision(dist::ExponentialLinearQuadratic)    = mean_precision(dist)[2]
weightedmean(dist::ExponentialLinearQuadratic) = weightedmean_precision(dist)[1]

prod_analytical_rule(::Type{ <: UnivariateNormalDistributionsFamily }, ::Type{ <: ExponentialLinearQuadratic }) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::UnivariateNormalDistributionsFamily, right::ExponentialLinearQuadratic)
    mean, variance = approximate_meancov(right.approximation, (z) -> pdf(right, z), left)
    return NormalMeanVariance(mean, variance)
end


