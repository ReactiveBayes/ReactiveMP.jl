"""
    GCV

The Gaussian controlled variance node, `y ~ N(x, exp(κz + ω))`: the variance of `y` about `x`
is `exp(κz + ω)`, so its precision is `exp(-(κz + ω))`. Its rules run under its own algorithm,
[`GCVApproximation`](@ref), and follow the factorisation: `q(y, x) q(z) q(κ) q(ω)` or mean-field.
"""
struct GCV end

"""
    GCVApproximation(; method = GaussHermiteCubature(20))

[`GCV`](@ref)'s algorithm: `method` is the cubature of the [`ExponentialLinearQuadratic`](@ref)
messages towards `z`, `κ` and `ω`. v6 called it `GCVMetadata`.
"""
struct GCVApproximation{M <: AbstractApproximationMethod} <: AbstractAlgorithm
    method::M
end

GCVApproximation(; method = GaussHermiteCubature(20)) = GCVApproximation(method)

@define_factor_node(node = GCV, type = Stochastic, interfaces = [:y, :x, :z, :κ, :ω], algorithm = GCVApproximation)

# log ⟨e^{-(κz + ω)}⟩, the log of the effective noise precision, as the sum of
#
#     log A = log ⟨e^{-ω}⟩  = -⟨ω⟩ + Var(ω) / 2              (exact, a lognormal's mean)
#     log B ≈ log ⟨e^{-κz}⟩ = -⟨κ⟩⟨z⟩ + Var(κz) / 2          (κz treated as normal)
#
# with Var(κz) = ⟨κ⟩² Var(z) + ⟨z⟩² Var(κ) + Var(κ) Var(z) for independent κ and z. Summing the
# exponents rather than forming A ⋅ B avoids a NaN where one overflows and the other underflows.
function log_noise_precision(q_z, q_κ, q_ω)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)
    ksi = κ_mean^2 * z_var + z_mean^2 * κ_var + κ_var * z_var
    return (-ω_mean + ω_var / 2) + (-κ_mean * z_mean + ksi / 2)
end

# ⟨(y - x)²⟩, under a joint q(y, x) or independent q(y) and q(x).
function expected_square_difference(q_y_x)
    m, V = mean_cov(q_y_x)
    return @inbounds (m[1] - m[2])^2 + V[1, 1] + V[2, 2] - V[1, 2] - V[2, 1]
end

function expected_square_difference(q_y, q_x)
    y_mean, y_var = mean_var(q_y)
    x_mean, x_var = mean_var(q_x)
    return (y_mean - x_mean)^2 + y_var + x_var
end

# ½[log 2π + ⟨κz + ω⟩ + ⟨(y - x)²⟩ ⟨e^{-(κz + ω)}⟩].
gcv_energy(psi, q_z, q_κ, q_ω) = (log2π + (mean(q_z) * mean(q_κ) + mean(q_ω)) + psi * exp(log_noise_precision(q_z, q_κ, q_ω))) / 2

@define_average_energy(
    node = GCV, args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:z]::NormalDistributionsFamily, q[:κ]::Any, q[:ω]::Any),
    body = (args) -> gcv_energy(expected_square_difference(args.q[:y, :x]), args.q[:z], args.q[:κ], args.q[:ω]),
)

@define_average_energy(
    node = GCV, args = (q[:y]::NormalDistributionsFamily, q[:x]::NormalDistributionsFamily, q[:z]::NormalDistributionsFamily, q[:κ]::Any, q[:ω]::Any),
    body = (args) -> gcv_energy(expected_square_difference(args.q[:y], args.q[:x]), args.q[:z], args.q[:κ], args.q[:ω]),
)
