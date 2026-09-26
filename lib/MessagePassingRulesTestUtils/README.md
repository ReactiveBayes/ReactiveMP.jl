# MessagePassingRulesTestUtils

Tools for testing message passing rules on their own, without a graph, for packages that define
nodes and rules with MessagePassingRulesBase:

- tables of cases for message rules, marginal rules and average energies, each case also run
  with inputs of other float types, through `rule!` for an in-place rule and on a poisoned
  scratch;
- verification of a message rule, and its log scale, against the node's log-density,
  integrated numerically;
- derivative checks, ForwardDiff through a rule against finite differences;
- the rule-coverage gate, which fails a suite when a rule has no test;
- comparison with a reference implementation, and recorded engine trajectories for whole
  inference runs.

```julia
using MessagePassingRulesTestUtils

@test_message_update_rule(
    node = NormalMeanVariance, target = :out,
    cases = [(m = (μ = PointMass(1.0), v = PointMass(2.0)),) => NormalMeanVariance(1.0, 2.0)],
)
@verify_message_update_rule(node = NormalMeanVariance, target = :out, m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))

# in test/runtests.jl, after an unfiltered run of the suite
@test isempty(check_rule_coverage(MyRules))
```

- Documentation: `make docs-testutils` from the repository root builds it into `docs/build`
  (after `make docs-base`, whose site it links to); it will be published at
  <https://reactivebayes.github.io/MessagePassingRulesTestUtils.jl/dev/>.
- Tests: `make test-testutils`.
- Depends on MessagePassingRulesBase, BayesBase, Distributions, ForwardDiff and HCubature, and
  on no engine. A test dependency: nothing at run time needs it. Julia 1.11 or later. MIT
  licence.
