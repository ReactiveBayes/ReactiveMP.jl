"""
    SoftDotMessagePassingRules

The SoftDot node, `y ~ N(θᵀx, γ⁻¹)`: the dot product of `θ` and `x` softened by Gaussian noise
of precision `γ`, with its variational rules. Where the deterministic dot product node, `dot`
in StandardMessagePassingRules, has no closed-form messages for two Gaussian factors, SoftDot's
are all in closed form. It serves Bayesian linear regression, `y ~ N(θᵀx, γ⁻¹)` with the
regressors `x` known, and bilinear models with both factors unknown.

- [`SoftDot`](@ref), the node, and its alias [`softdot`](@ref). It runs under
  [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), which a model need not
  name.
- Its rules are variational, under the mean-field `q(y)q(θ)q(x)q(γ)` or the structured
  `q(y, x)q(θ)q(γ)`, with an average energy for each. There are no belief-propagation rules.

# Examples

```jldoctest
julia> using MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> q_θ = MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0]);

julia> q_x = MvNormalMeanCovariance([3.0, 1.0], [1.0 0.0; 0.0 1.0]);

julia> result = @call_message_update_rule(node = SoftDot, target = :y, q = (θ = q_θ, x = q_x, γ = GammaShapeRate(2.0, 1.0)));

julia> all(mean_precision(getresult(result)) .≈ (5.0, 2.0))
true
```
"""
module SoftDotMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using LinearAlgebra: dot
using StatsFuns: log2π
import StandardMessagePassingRules

export SoftDot, softdot

include("helpers.jl")
include("node.jl")
include("rules.jl")

end
