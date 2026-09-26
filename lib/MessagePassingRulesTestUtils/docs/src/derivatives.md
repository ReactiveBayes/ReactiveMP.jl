# Derivatives

A model that is differentiated, to fit hyperparameters or to run a gradient-based sampler, needs
derivatives to pass through its rules. [`@test_rule_derivatives`](@ref) builds the rule's inputs
from a parameter `θ`, reduces the output to a number with `summary`, and compares ForwardDiff's
derivative with a central finite difference:

```julia
@test_rule_derivatives(
    node = NormalMeanVariance, target = :out,
    inputs = θ -> (m = (μ = NormalMeanVariance(θ, 1.0), v = PointMass(θ^2)),),
    at = 1.5, summary = d -> mean(d) + var(d),
)
```

A rule that converts its inputs to `Float64`, or strips the dual part of a number, gives a wrong
or zero derivative and fails. An in-place rule is checked through `rule` and through `rule!`.

## API

```@docs
@test_rule_derivatives
test_rule_derivatives
```
