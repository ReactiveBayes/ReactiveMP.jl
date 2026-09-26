# FlowMessagePassingRules

The `Flow` node for the ReactiveMP ecosystem: the deterministic node `out = f(in)` for an
invertible normalising-flow model `f`, with the models it is built from. A model is a tuple of
layers, `FlowModel`, given its parameters by `compile`: additive coupling layers holding planar
or radial flows, and permutation layers. The node's rules push a multivariate normal through the
compiled model, forwards or backwards, by linearisation or by the unscented transform, under
`FlowApproximation`, which a model must give.

```julia
using FlowMessagePassingRules, MessagePassingRulesBase, ExponentialFamily

model = compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow()),)))
m_in = MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0])
result = @call_message_update_rule(node = Flow, target = :out, m = (in = m_in,), algorithm = FlowApproximation(model))
getresult(result)   # an MvNormalMeanCovariance, the linearised image of m_in
```

- Documentation: `make docs-flow` from the repository root builds it into `docs/build`; it will
  be published at <https://reactivebayes.github.io/FlowMessagePassingRules.jl/dev/>.
- Tests: `make test-flow`.
- Depends on MessagePassingRulesBase and MessagePassingRulesApproximations, with BayesBase,
  ExponentialFamily, Distributions, LinearAlgebra, Random and TupleTools. Julia 1.11 or later.
  MIT licence.
