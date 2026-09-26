# StandardMessagePassingRules

The message passing rules of the standard factor nodes, written with MessagePassingRulesBase:

- **distributions**: the univariate and multivariate normals in every parametrisation,
  `GammaShapeRate`, `Gamma`, `GammaInverse`, `Beta`, `Bernoulli`, `Categorical`, `Dirichlet`,
  `DirichletCollection`, `Poisson`, `Uniform`, `Wishart`, `InverseWishart`, `MatrixNormal`,
  `MatrixNormalWishart`, `MvNormalGamma`, `MvNormalWishart` and `HalfNormal`;
- **arithmetic**: `+`, `-`, `*` and `dot`, the functions being the nodes;
- **logic**: `AND`, `OR`, `NOT` and `IMPLY` over Bernoulli variables;
- **mixtures**: `NormalMixture`, `GammaMixture` and `Mixture`;
- **helpers**: `StandaloneDistribution`, `Uninformative` and `diageye`.

Most nodes are ExponentialFamily's, Distributions' and Base's own types and functions; loading
the package is enough for an engine to find their rules. The distribution nodes' rules cover
belief propagation, mean field and structured factorisations under one default algorithm.

```julia
using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

result = @call_message_update_rule(
    node = NormalMeanVariance, target = :out,
    m = (μ = NormalMeanVariance(0.0, 1.0), v = PointMass(2.0)),
)
getresult(result)                                          # NormalMeanVariance(0.0, 3.0)

MessagePassingRulesBase.rule_coverage(NormalMeanVariance)  # which rules exist
```

- Documentation: `make docs-standard` from the repository root builds it into `docs/build`; it
  will be published at <https://reactivebayes.github.io/StandardMessagePassingRules.jl/dev/>.
- Tests: `make test-standard`, which ends with a rule-coverage gate: every rule must be
  selected by some test.
- Depends on MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions and a few
  numerical packages (FastCholesky, MatrixCorrectionTools, SpecialFunctions, StatsFuns,
  LogExpFunctions, DomainSets), not on ReactiveMP. Julia 1.11 or later. MIT licence.
