# DiscreteTransitionMessagePassingRules

The `DiscreteTransition` node, `out ~ Categorical(A[:, in, T1, …, Tn])`: a transition between
categoricals through a tensor `A` of probabilities, conditioned on any number of categoricals
`T`, with `A` known (a `PointMass`) or learned (a `DirichletCollection`). It is the transition and
emission node of hidden Markov models. As a tensor node, each of its rules is one tensor
contraction, written once for belief propagation, mean-field and structured factorisations and
any number of `T`s. It runs under the default algorithm and has an average energy.

```julia
using DiscreteTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily

result = @call_message_update_rule(
    node = DiscreteTransition, target = :out,
    m = (in = Categorical([0.5, 0.5]),), q = (a = PointMass([0.9 0.2; 0.1 0.8]),),
)
getresult(result)   # Categorical([0.55, 0.45])
```

- Documentation: `make docs-discrete-transition` from the repository root builds it into
  `docs/build`; it will be published at
  <https://reactivebayes.github.io/DiscreteTransitionMessagePassingRules.jl/dev/>.
- Tests: `make test-discrete-transition`.
- Depends on MessagePassingRulesBase, BayesBase, ExponentialFamily and Distributions.
- MIT licence.
