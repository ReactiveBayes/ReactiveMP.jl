# AutoregressiveMessagePassingRules

The message passing rules of the autoregressive nodes, for Bayesian autoregressive processes of
order `p`, `yₜ ~ N(θᵀxₜ, γ⁻¹)` with `xₜ = (yₜ₋₁, …, yₜ₋ₚ)`, whose states, coefficients and
noise precision are inferred together:

- `AR` (alias `Autoregressive`): the coefficients `θ` and the precision `γ` on edges of their
  own;
- `ConjugateAR`: `(θ, γ)` joint on one normal-gamma edge `w`, for a conjugate parameter update.

Their rules are variational and run under `ARVMP(form, order, stype)`, which a model must give;
`stype` is `ARsafe()` or `ARunsafe()`.

```julia
using AutoregressiveMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

result = @call_message_update_rule(
    node = AR, target = :y, algorithm = ARVMP(Univariate, 1, ARsafe()),
    m = (x = NormalMeanVariance(1.0, 1.0),),
    q = (θ = NormalMeanVariance(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0)),
)
getresult(result)   # NormalMeanVariance(0.5, 1.5)
```

- Documentation: `make docs-autoregressive` from the repository root builds it into
  `docs/build`; it will be published at <https://reactivebayes.github.io/AutoregressiveMessagePassingRules.jl/dev/>.
- Tests: `make test-autoregressive`.
- Depends on MessagePassingRulesBase and StandardMessagePassingRules, on BayesBase,
  ExponentialFamily and Distributions for the distributions, and on FastCholesky, LinearAlgebra
  and StatsFuns. Julia 1.11 or later. MIT licence.
