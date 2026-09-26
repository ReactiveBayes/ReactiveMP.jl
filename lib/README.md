# `lib/` — the message passing rule packages

The rule system, its test tooling, the numerics the rules share, and the rules themselves,
one package per group of nodes. Each package has its own test suite. The engine in `src/`
depends on `MessagePassingRulesBase` only; it finds the rules of every loaded rule package
through it.

| package | role |
|---|---|
| `MessagePassingRulesBase` | the rule system: the macros that declare nodes, rules, marginal rules, average energies and dependencies; targets, algorithms, the argument and annotation containers, the rule context, the registry and rule lookup, rule fallbacks, and the interactive surface (`@call_*`, `@which_*`, `list_rules`, `rule_coverage`) |
| `MessagePassingRulesTestUtils` | test tooling, used through `[extras]`: table tests (`@test_message_update_rule`), verification against a node's definition, derivative checks, the rule-coverage gate, comparison with a reference implementation, and engine trajectories |
| `StandardMessagePassingRules` | the standard nodes: distributions, arithmetic, logic and the mixtures |
| `MessagePassingRulesApproximations` | numerics over means and covariances: `Unscented`, `Linearization`, Gauss–Hermite cubature and `smoothRTS` |
| `DeltaMessagePassingRules` | the Delta node, `DeltaFn{F}`, under `DeltaApproximation(; method, inverse)` with `Unscented()`, `Linearization()` or, through an extension on ExponentialFamilyProjection, `CVIProjection()` |
| `GaussianCouplingMessagePassingRules` | the GaussianCoupling node, the edge potential of Gaussian belief propagation |
| `ProbitMessagePassingRules` | the Probit node and its algorithm `ProbitEP`, expectation propagation |
| `GCVMessagePassingRules` | the GCV node, its `ExponentialLinearQuadratic`, and the normal nodes' rules for it |
| `AutoregressiveMessagePassingRules` | the AR and ConjugateAR nodes under their algorithm `ARVMP` |
| `SoftDotMessagePassingRules` | the SoftDot node, independent of the autoregressive package |
| `ContinuousTransitionMessagePassingRules` | the ContinuousTransition node under `CTVMP(f)`; its rule towards `a` extends the default scheme with `q(a)` |
| `PolyaMessagePassingRules` | BinomialPolya and MultinomialPolya, Pólya-Gamma augmented; **GPL-3** through PolyaGammaHybridSamplers, the only package that is |
| `BIFMMessagePassingRules` | BIFM and BIFMHelper under `BIFMSmoother(A, B, C)`: stateless rules, no free energy |
| `FlowMessagePassingRules` | Flow under `FlowApproximation(model; method)`, the flow models and layers, and `PermutationMatrix` |
| `DiscreteTransitionMessagePassingRules` | DiscreteTransition as a tensor node: one rule per target, over whatever inputs the factorisation delivers |

## Documentation

Each package has a `README.md` and its own documentation site, `docs/` beside its `src/`, built
with `make docs-<package>` (the test target's name, `docs-base`, `docs-standard`, …) into
`docs/build`; `make docs-all` builds every site in dependency order, as CI does. The sites link
to each other with DocumenterInterLinks `@extref`, against their planned addresses,
`https://reactivebayes.github.io/<Package>.jl/dev/`, and each sibling's local inventory until they
are published with the repository split.

## Dependency constraints

- `MessagePassingRulesBase` never depends on `ExponentialFamily`: the distribution machinery
  it needs is BayesBase's, and anything missing is added to BayesBase, in non-breaking
  releases, since `compat/v6-comparison` resolves against them.
- `MessagePassingRulesApproximations` depends on no sibling and on no distribution package:
  approximating an integral is numerics, choosing which rules run is an algorithm.
- No rule package depends on `ReactiveMP` or on `MessagePassingRulesTestUtils`; each
  package's `quality:closure` test checks its dependency closure.

## Wiring

The packages are unregistered, so a package that depends on a sibling lists it in `[deps]`
and in `[sources]`, with its relative path. A test-only sibling, such as
`MessagePassingRulesTestUtils`, goes in `[extras]` and `[sources]`. `Pkg.test()` then works
from each package's own project, and so does its `make` target:

```bash
make test-base test-testutils test-standard test-approximations test-delta \
     test-gaussian-coupling test-probit test-gcv test-autoregressive test-softdot \
     test-continuous-transition test-polya test-bifm test-flow test-discrete-transition
```

Each target takes `test_args` as the root's `make test` does. Items tagged `:slow` are skipped
unless `TEST_ALL=true`, which CI sets. `[sources]` needs Julia 1.11 or later; the packages are
developed and tested on Julia 1.13. No `Manifest.toml` under `lib/` is committed.
