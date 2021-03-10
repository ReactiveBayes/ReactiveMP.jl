export GCV, GCVMetadata

import StatsFuns: log2π

const DefaultGCVNodeVarianceApproximation = GaussHermiteCubature(20)

struct GCVMetadata{ A <: AbstractApproximationMethod }
    approximation :: A
end

struct GCV end

get_variance_approximation(::Type{ <: GCV }, meta::GCVMetadata) = meta.approximation
get_variance_approximation(::Type{ <: GCV }, meta::Nothing)     = DefaultGCVNodeVarianceApproximation

@node GCV Stochastic [ y, x, z, κ, ω ]

@average_energy GCV (q_y_x::MultivariateNormalDistributionsFamily, q_z::NormalDistributionsFamily, q_κ::Any, q_ω::Any) = begin
    m, c = mean(q_y_x), cov(q_y_x)

    ksi = (mean(q_κ) ^ 2) * var(q_z) + (mean(q_z) ^ 2) * var(q_κ) + var(q_κ) * var(q_z)
    psi = (m[2] - m[1]) ^ 2 + c[1, 1] + c[2, 2] - c[1, 2] - c[2, 1]
    A = exp(-mean(q_ω) + var(q_ω) / 2)
    B = exp(-mean(q_κ) * mean(q_z) + ksi / 2)

    0.5 * log2π + 0.5 * (mean(q_z) * mean(q_κ) + mean(q_ω)) + 0.5 * (psi * A * B)
end