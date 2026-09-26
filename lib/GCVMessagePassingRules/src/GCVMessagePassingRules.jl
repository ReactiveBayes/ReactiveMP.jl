"""
    GCVMessagePassingRules

The Gaussian controlled variance node, [`GCV`](@ref): a normal whose log-variance is linear in
other variables of the model,

```math
p(y \\mid x, z, κ, ω) = \\mathcal{N}\\bigl(y \\mid x, \\exp(κ z + ω)\\bigr),
```

the building block of hierarchical Gaussian filters, where one layer's state sets the volatility
of the layer below.

- [`GCV`](@ref): the node, with interfaces `y`, `x`, `z`, `κ` and `ω`. Its rules are
  variational, under the structured factorisation `q(y, x) q(z) q(κ) q(ω)` or the mean field;
  every rule reads the marginals of `z`, `κ` and `ω`.
- [`GCVApproximation`](@ref): its algorithm, which carries the Gauss–Hermite cubature of the
  messages towards `z`, `κ` and `ω`.
- [`ExponentialLinearQuadratic`](@ref): the density of those messages, which has no closed-form
  moments.

The package also adds rules to `NormalMeanVariance` and `NormalMeanPrecision` (their nodes are in
StandardMessagePassingRules) for an [`ExponentialLinearQuadratic`](@ref) message on `out`, so that a
GCV's `z` or `ω` can be the output of a normal node.

# Examples

```jldoctest
julia> using MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> q_z, q_κ, q_ω = PointMass(0.0), PointMass(1.0), PointMass(0.0);  # a noise variance exp(0) = 1

julia> result = @call_message_update_rule(
           node = GCV, target = :y, m = (x = NormalMeanVariance(1.0, 2.0),), q = (z = q_z, κ = q_κ, ω = q_ω),
       );

julia> mean_var(getresult(result)) .≈ (1.0, 3.0)
(true, true)
```
"""
module GCVMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm
using MessagePassingRulesApproximations: AbstractApproximationMethod, GaussHermiteCubature, approximate_meancov
using StatsFuns: log2π
import StandardMessagePassingRules

export GCV, GCVApproximation, ExponentialLinearQuadratic

include("exponential_linear_quadratic.jl")
include("node.jl")
include("rules.jl")
include("gaussian_extension.jl")

end
