"""
    GCV

The Gaussian controlled variance node, a normal whose log-variance is linear in its inputs:

```math
p(y \\mid x, z, κ, ω) = \\mathcal{N}\\bigl(y \\mid x, \\exp(κ z + ω)\\bigr).
```

The variance of `y` about `x` is `exp(κz + ω)`, so its precision is `exp(-(κz + ω))`. It is
stochastic, with the interfaces, in order:

- `y`: the output, normal about `x`;
- `x`: the mean of `y`;
- `z`: the variable that controls the log-variance, scaled by `κ`;
- `κ`: the coupling of `z` to the log-variance;
- `ω`: the offset of the log-variance.

Its rules run under its own algorithm, [`GCVApproximation`](@ref), whose default instance the
node declares, so a model need not name it. They are variational and read `q(z)`, `q(κ)` and
`q(ω)` always, under one of two factorisations:

- structured, `q(y, x) q(z) q(κ) q(ω)`: the rules towards `y` and `x` take the message on the
  other one, the rules towards `z`, `κ` and `ω` the joint `q(y, x)`, and there is a marginal rule
  for `q(y, x)`;
- mean field, `q(y) q(x) q(z) q(κ) q(ω)`: every rule takes marginals.

Towards `y` and `x` a rule returns a `NormalMeanVariance`; towards `z`, `κ` and `ω` an
[`ExponentialLinearQuadratic`](@ref). The expectation `⟨exp(-κz)⟩` treats `κz` as normal, which
is exact only when `κ` or `z` is a point mass; `⟨exp(-ω)⟩` is exact for a normal `q(ω)`.

The average energy,

```math
\\tfrac12\\bigl[\\log 2π + ⟨κ⟩⟨z⟩ + ⟨ω⟩ + ⟨(y - x)^2⟩ ⟨e^{-(κz + ω)}⟩\\bigr],
```

is defined for both factorisations, with a multivariate normal `q(y, x)` or normal `q(y)` and
`q(x)`, and a normal `q(z)`; `q(κ)` and `q(ω)` need only a mean and a variance.

# Limitations

- There are no belief-propagation rules: every rule needs the marginals of `z`, `κ` and `ω`.
- The messages on `y` and `x` must be univariate normals or
  [`ExponentialLinearQuadratic`](@ref)s; the node is univariate.
- The rules declare no log scale.

See also [`GCVApproximation`](@ref), [`ExponentialLinearQuadratic`](@ref).
"""
struct GCV end

"""
    GCVApproximation(; method = GaussHermiteCubature(20))
    GCVApproximation(method)

The algorithm of [`GCV`](@ref)'s rules: it carries the cubature that computes the moments of the
[`ExponentialLinearQuadratic`](@ref) messages towards `z`, `κ` and `ω`. The rules towards `y` and
`x` do not use it.

# Keywords

- `method`: a [`GaussHermiteCubature`](@extref MessagePassingRulesApproximations.GaussHermiteCubature),
  of any number of points. Default `GaussHermiteCubature(20)`. More points give more accurate
  moments at a higher cost.

The node's rules are declared for the default instance's type, so a method other than a
[`GaussHermiteCubature`](@extref MessagePassingRulesApproximations.GaussHermiteCubature) finds no
rule. A model names this algorithm only to change the number of points.

# Examples

```jldoctest
julia> using MessagePassingRulesApproximations: GaussHermiteCubature

julia> GCVApproximation(method = GaussHermiteCubature(32)) isa GCVApproximation
true
```
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
