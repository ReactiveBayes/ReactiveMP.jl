@doc raw"""
    AutoregressiveMessagePassingRules

The autoregressive nodes, for Bayesian autoregressive processes of order `p`,
`yₜ ~ N(θᵀxₜ, γ⁻¹)` with `xₜ = (yₜ₋₁, …, yₜ₋ₚ)`:

- [`AR`](@ref) (alias [`Autoregressive`](@ref)): the coefficients `θ` and the precision `γ` on
  edges of their own;
- [`ConjugateAR`](@ref): `(θ, γ)` joint on one `MvNormalGamma` edge `w`, which makes the
  parameter update conjugate.

Their rules run under [`ARVMP`](@ref)`(form, order, stype)`, which the model must give: the
nodes declare no algorithm, and under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm) no rule exists. `stype`
is [`ARsafe`](@ref)`()` or [`ARunsafe`](@ref)`()`, the way the joint `q(y, x)` is formed. The
rules are variational, under the structured factorisation `q(y, x) q(θ) q(γ)` (`q(y, x) q(w)`)
or the mean field, and both nodes have an average energy.

# Examples

```jldoctest
julia> algorithm = ARVMP(Multivariate, 2, ARsafe());

julia> result = @call_message_update_rule(
           node = AR, target = :γ, algorithm = algorithm,
           clusters = ((:y, :x) => MvNormalMeanCovariance(ones(4), [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]),),
           q = (θ = MvNormalMeanCovariance([0.5, 0.25], [1.0 0.0; 0.0 1.0]),),
       );

julia> shape(getresult(result)) ≈ 1.5 && rate(getresult(result)) ≈ 2.6875
true
```
"""
module AutoregressiveMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm
using BayesBase: huge, promote_paramfloattype, promote_variate_type
using Distributions: VariateForm
using FastCholesky: cholinv
using StatsFuns: log2π
import LinearAlgebra
using LinearAlgebra: dot, pinv
import StandardMessagePassingRules

export AR, Autoregressive, ConjugateAR, ARVMP, ARsafe, ARunsafe

include("algebra/companion_matrix.jl")
include("algebra/standard_basis_vector.jl")
include("algebra/ar_matrices.jl")
include("autoregressive.jl")
include("conjugate_autoregressive.jl")

end
