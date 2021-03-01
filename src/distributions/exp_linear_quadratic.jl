export ExponentialLinearQuadratic

import Distributions: pdf, logpdf, ContinuousUnivariateDistribution

struct ExponentialLinearQuadratic{ A <: AbstractApproximationMethod, T <: Real } <: ContinuousUnivariateDistribution
    approximation :: A
    a :: T
    b :: T
    c :: T
    d :: T
end

ExponentialLinearQuadratic(a::Real, b::Real, c::Real, d::Real)             = ExponentialLinearQuadratic(promote(a, b, c, d)...)
ExponentialLinearQuadratic(a::Integer, b::Integer, c::Integer, d::Integer) = ExponentialLinearQuadratic(float(a), float(b), float(c), float(d))

Distributions.pdf(dist::ExponentialLinearQuadratic, x::Real)    = exp(logpdf(dist, x))
Distributions.logpdf(dist::ExponentialLinearQuadratic, x::Real) = -0.5 * (dist.a * x + dist.b * exp(dist.c * x + dist.d * x ^ 2 / 2.0))

function prod(::ProdPreserveParametrisation, left::UnivariateNormalDistributionsFamily, right::ExponentialLinearQuadratic)
    mean, variance = approximate_meancov(right.approximation, (z) -> pdf(right, z), left)
    return NormalMeanVariance(mean, variance)
end


