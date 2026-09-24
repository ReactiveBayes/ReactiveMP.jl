"""
    ExponentialLinearQuadratic(approximation, a, b, c, d)

The density `∝ exp(-(a x + b exp(c x + d x² / 2)) / 2)`, the form of the GCV node's messages
towards `z`, `κ` and `ω`. It has no closed-form moments: `approximation`, a cubature such as
`GaussHermiteCubature(20)`, computes them. Its product with a univariate normal is a
`NormalMeanVariance` of the product's moments.
"""
struct ExponentialLinearQuadratic{A <: AbstractApproximationMethod, T <: Real} <: ContinuousUnivariateDistribution
    approximation::A
    a::T
    b::T
    c::T
    d::T
end

ExponentialLinearQuadratic(approximation, a::Real, b::Real, c::Real, d::Real) = ExponentialLinearQuadratic(approximation, promote(a, b, c, d)...)
ExponentialLinearQuadratic(approximation::AbstractApproximationMethod, a::T, b::T, c::T, d::T) where {T <: Integer} = ExponentialLinearQuadratic(approximation, float(a), float(b), float(c), float(d))

Base.eltype(::Type{<:ExponentialLinearQuadratic{A, T}}) where {A, T} = T
Distributions.params(dist::ExponentialLinearQuadratic) = (dist.a, dist.b, dist.c, dist.d)
Base.precision(dist::ExponentialLinearQuadratic) = mean_invcov(dist)[2]

# The moments of the density, as those of `exp(-(a x - x² + b exp(c x + d x²/2)) / 2) N(x | 0, 1)`.
function BayesBase.mean_var(dist::ExponentialLinearQuadratic)
    adjusted_pdf = let a = dist.a, b = dist.b, c = dist.c, d = dist.d
        x -> exp(-(a * x - x^2 + b * exp(c * x + d * x^2 / 2)) / 2)
    end
    return approximate_meancov(dist.approximation, adjusted_pdf, zero(eltype(dist)), one(eltype(dist)))
end

BayesBase.mean_cov(dist::ExponentialLinearQuadratic) = mean_var(dist)
BayesBase.mean_invcov(dist::ExponentialLinearQuadratic) = mean_cov(dist) .|> (identity, inv)
BayesBase.mean_std(dist::ExponentialLinearQuadratic) = mean_var(dist) .|> (identity, sqrt)
BayesBase.weightedmean_cov(dist::ExponentialLinearQuadratic) = weightedmean_var(dist)
BayesBase.weightedmean_std(dist::ExponentialLinearQuadratic) = weightedmean_var(dist) .|> (identity, sqrt)

function BayesBase.weightedmean_var(dist::ExponentialLinearQuadratic)
    m, v = mean_cov(dist)
    return (inv(v) * m, v)
end

function BayesBase.weightedmean_invcov(dist::ExponentialLinearQuadratic)
    m, w = mean_invcov(dist)
    return (w * m, w)
end

BayesBase.pdf(dist::ExponentialLinearQuadratic, x::Real) = exp(logpdf(dist, x))
BayesBase.logpdf(dist::ExponentialLinearQuadratic, x::Real) = -(dist.a * x + dist.b * exp(dist.c * x + dist.d * x^2 / 2)) / 2
BayesBase.mean(dist::ExponentialLinearQuadratic) = mean_var(dist)[1]
BayesBase.var(dist::ExponentialLinearQuadratic) = mean_var(dist)[2]
BayesBase.std(dist::ExponentialLinearQuadratic) = mean_std(dist)[2]
BayesBase.cov(dist::ExponentialLinearQuadratic) = var(dist)
BayesBase.invcov(dist::ExponentialLinearQuadratic) = mean_invcov(dist)[2]
BayesBase.weightedmean(dist::ExponentialLinearQuadratic) = weightedmean_invcov(dist)[1]

BayesBase.default_prod_rule(::Type{<:UnivariateNormalDistributionsFamily}, ::Type{<:ExponentialLinearQuadratic}) = PreserveTypeProd(NormalMeanVariance)

function BayesBase.prod(::PreserveTypeProd{NormalMeanVariance}, left::UnivariateNormalDistributionsFamily, right::ExponentialLinearQuadratic)
    m, v = approximate_meancov(right.approximation, z -> pdf(right, z), mean(left), var(left))
    return NormalMeanVariance(m, v)
end

# A normal or the moments-only density, which the rules towards `y` and `x` treat alike.
const UniNormalOrExpLinQuad = Union{UnivariateGaussianDistributionsFamily, ExponentialLinearQuadratic}
