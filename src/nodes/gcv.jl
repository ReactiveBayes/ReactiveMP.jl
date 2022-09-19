export GCV, GCVMetadata

import StatsFuns: log2π

const UniNormalOrExpLinQuad = Union{UnivariateGaussianDistributionsFamily, ExponentialLinearQuadratic}

struct GCVMetadata{A <: AbstractApproximationMethod}
    approximation::A
end

get_approximation(meta::GCVMetadata) = meta.approximation

struct GCV end

@node GCV Stochastic [y, x, z, κ, ω]

const DefaultGCVNodeMetadata = GCVMetadata(GaussHermiteCubature(20))

default_meta(::Type{GCV}) = DefaultGCVNodeMetadata

@average_energy GCV (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_z::NormalDistributionsFamily,
    q_κ::Any,
    q_ω::Any,
    meta::Union{<:GCVMetadata, Nothing}
) =
    begin
        y_x_mean, y_x_cov = mean_cov(q_y_x)
        z_mean, z_var     = mean_var(q_z)
        κ_mean, κ_var     = mean_var(q_κ)
        ω_mean, ω_var     = mean_var(q_ω)

        ksi = (κ_mean^2) * z_var + κ_var * ((z_mean^2) + z_var)
        psi = @inbounds (y_x_mean[2] - y_x_mean[1])^2 + y_x_cov[1, 1] + y_x_cov[2, 2] - y_x_cov[1, 2] - y_x_cov[2, 1]
        A = exp(-ω_mean + ω_var / 2)
        B = exp(-κ_mean * z_mean + ksi / 2)

        0.5 * log2π + 0.5 * (z_mean * κ_mean + ω_mean) + 0.5 * (psi * A * B)
    end

@average_energy GCV (
    q_y::NormalDistributionsFamily,
    q_x::NormalDistributionsFamily,
    q_z::NormalDistributionsFamily,
    q_κ::Any,
    q_ω::Any
) = begin
    y_mean, y_var = mean_var(q_y)
    x_mean, x_var = mean_var(q_x)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    ksi = (κ_mean^2) * z_var + (z_mean^2) * κ_var + κ_var * z_var
    psi = (y_mean - x_mean)^2 + y_var + x_var
    A = exp(-ω_mean + ω_var / 2)
    B = exp(-κ_mean * z_mean + ksi / 2)

    0.5 * log2π + 0.5 * (z_mean * κ_mean + ω_mean) + 0.5 * (psi * A * B)
end
