export ExponentialLinearQuadratic

import Distributions: pdf, logpdf, ContinuousUnivariateDistribution

struct ExponentialLinearQuadratic{T <: Real} <: ContinuousUnivariateDistribution
    a :: T
    b :: T
    c :: T
    d :: T
end

ExponentialLinearQuadratic(a::Real, b::Real, c::Real, d::Real)             = ExponentialLinearQuadratic(promote(a, b, c, d)...)
ExponentialLinearQuadratic(a::Integer, b::Integer, c::Integer, d::Integer) = ExponentialLinearQuadratic(float(a), float(b), float(c), float(d))

Distributions.pdf(dist::ExponentialLinearQuadratic, x::Real)    = exp(logpdf(dist, x))
Distributions.logpdf(dist::ExponentialLinearQuadratic, x::Real) = -0.5 * (dist.a * x + dist.b * exp(dist.c * x + dist.d * x ^ 2 / 2.0))

function prod(::ProdPreserveParametrisation, left::NormalMeanVariance{T}, right::ExponentialLinearQuadratic{T}) where T
    mean, var = approximate_meancov(ghcubature(5), (z) -> pdf(right, z), left)
    return NormalMeanVariance(mean, var)
end