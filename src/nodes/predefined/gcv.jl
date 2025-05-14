export GCV, GCVMetadata

import StatsFuns: log2π

struct ExponentialLinearQuadratic{A <: AbstractApproximationMethod, T <: Real} <: ContinuousUnivariateDistribution
    approximation::A
    a::T
    b::T
    c::T
    d::T
end

ExponentialLinearQuadratic(approximation, a::Real, b::Real, c::Real, d::Real)             = ExponentialLinearQuadratic(approximation, promote(a, b, c, d)...)
ExponentialLinearQuadratic(approximation, a::Integer, b::Integer, c::Integer, d::Integer) = ExponentialLinearQuadratic(approximation, float(a), float(b), float(c), float(d))

Base.eltype(::Type{<:ExponentialLinearQuadratic{A, T}}) where {A, T} = T
Base.eltype(::ExponentialLinearQuadratic{A, T}) where {A, T}         = T

Base.precision(dist::ExponentialLinearQuadratic) = mean_invcov(dist)[2]

BayesBase.mean_cov(dist::ExponentialLinearQuadratic) = mean_var(dist)

function BayesBase.mean_var(dist::ExponentialLinearQuadratic)
    # This is equivalent to (x) -> pdf(dist, x) * exp(x^2 / 2)
    adjusted_pdf = let a = dist.a, b = dist.b, c = dist.c, d = dist.d
        (x) -> exp(-(a * x - x^2 + b * exp(c * x + d * x^2 / 2)) / 2)
    end
    return approximate_meancov(dist.approximation, adjusted_pdf, NormalMeanVariance(zero(eltype(dist)), one(eltype(dist))))
end

BayesBase.mean_invcov(dist::ExponentialLinearQuadratic)      = mean_cov(dist) .|> (identity, inv)
BayesBase.mean_std(dist::ExponentialLinearQuadratic)         = mean_var(dist) .|> (identity, sqrt)
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

BayesBase.pdf(dist::ExponentialLinearQuadratic, x::Real)    = exp(logpdf(dist, x))
BayesBase.logpdf(dist::ExponentialLinearQuadratic, x::Real) = -(dist.a * x + dist.b * exp(dist.c * x + dist.d * x^2 / 2)) / 2
BayesBase.mean(dist::ExponentialLinearQuadratic)            = mean_var(dist)[1]
BayesBase.var(dist::ExponentialLinearQuadratic)             = mean_var(dist)[2]
BayesBase.std(dist::ExponentialLinearQuadratic)             = mean_std(dist)[2]
BayesBase.cov(dist::ExponentialLinearQuadratic)             = var(dist)

BayesBase.invcov(dist::ExponentialLinearQuadratic)       = mean_invcov(dist)[2]
BayesBase.weightedmean(dist::ExponentialLinearQuadratic) = weightedmean_invcov(dist)[1]

BayesBase.default_prod_rule(::Type{<:UnivariateNormalDistributionsFamily}, ::Type{<:ExponentialLinearQuadratic}) = PreserveTypeProd(NormalMeanVariance)

function prod(::PreserveTypeProd{NormalMeanVariance}, left::UnivariateNormalDistributionsFamily, right::ExponentialLinearQuadratic)
    mean, variance = approximate_meancov(right.approximation, (z) -> pdf(right, z), left)
    return NormalMeanVariance(mean, variance)
end

const UniNormalOrExpLinQuad = Union{UnivariateGaussianDistributionsFamily, ExponentialLinearQuadratic}

struct GCVMetadata{A <: AbstractApproximationMethod}
    approximation::A
end

get_approximation(meta::GCVMetadata) = meta.approximation

struct GCV end

@node GCV Stochastic [y, x, z, κ, ω]

const DefaultGCVNodeMetadata = GCVMetadata(GaussHermiteCubature(20))

default_meta(::Type{GCV}) = DefaultGCVNodeMetadata

@average_energy GCV (q_y_x::MultivariateNormalDistributionsFamily, q_z::NormalDistributionsFamily, q_κ::Any, q_ω::Any, meta::Union{<:GCVMetadata, Nothing}) = begin
    y_x_mean, y_x_cov = mean_cov(q_y_x)
    z_mean, z_var     = mean_var(q_z)
    κ_mean, κ_var   = mean_var(q_κ)
    ω_mean, ω_var   = mean_var(q_ω)

    ksi = (κ_mean^2) * z_var + κ_var * ((z_mean^2) + z_var)
    psi = @inbounds (y_x_mean[2] - y_x_mean[1])^2 + y_x_cov[1, 1] + y_x_cov[2, 2] - y_x_cov[1, 2] - y_x_cov[2, 1]
    A = exp(-ω_mean + ω_var / 2)
    B = exp(-κ_mean * z_mean + ksi / 2)

    (log2π + (z_mean * κ_mean + ω_mean) + (psi * A * B)) / 2
end

@average_energy GCV (q_y::NormalDistributionsFamily, q_x::NormalDistributionsFamily, q_z::NormalDistributionsFamily, q_κ::Any, q_ω::Any, meta::Union{<:GCVMetadata, Nothing}) =
    begin
        y_mean, y_var = mean_var(q_y)
        x_mean, x_var = mean_var(q_x)
        z_mean, z_var = mean_var(q_z)
        κ_mean, κ_var = mean_var(q_κ)
        ω_mean, ω_var = mean_var(q_ω)

        ksi = (κ_mean^2) * z_var + (z_mean^2) * κ_var + κ_var * z_var
        psi = (y_mean - x_mean)^2 + y_var + x_var
        A = exp(-ω_mean + ω_var / 2)
        B = exp(-κ_mean * z_mean + ksi / 2)

        (log2π + (z_mean * κ_mean + ω_mean) + (psi * A * B)) / 2
    end
