# `lib/` — the message passing rule packages

The rule packages, one per group of nodes, each with its own test suite. The rule system they are
written with, [MessagePassingRulesBase](https://github.com/ReactiveBayes/MessagePassingRulesBase.jl),
the numerics they share,
[MessagePassingRulesApproximations](https://github.com/ReactiveBayes/MessagePassingRulesApproximations.jl),
and their test tooling,
[MessagePassingRulesTestUtils](https://github.com/ReactiveBayes/MessagePassingRulesTestUtils.jl),
have repositories of their own and are registered. The engine in `src/` depends on `MessagePassingRulesBase` only;
it finds the rules of every loaded rule package through it.

| package | role |
|---|---|
| `StandardMessagePassingRules` | the standard nodes: distributions, arithmetic, logic and the mixtures |
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
with `make docs-<package>` (the test target's name, `docs-standard`, `docs-delta`, …) into
`docs/build`; `make docs-all` builds every site in dependency order, as CI does. The sites link
to each other with DocumenterInterLinks `@extref`: to the registered packages' published sites,
and to the other packages' planned addresses, `https://reactivebayes.github.io/<Package>.jl/dev/`,
with each sibling's local inventory until they are published with the repository split.

## Dependency constraints

- `MessagePassingRulesBase` never depends on `ExponentialFamily`: the distribution machinery
  it needs is BayesBase's, and anything missing is added to BayesBase, in non-breaking
  releases, since `compat/v6-comparison` resolves against them.
- No rule package depends on `ReactiveMP` or on `MessagePassingRulesTestUtils`; each
  package's `quality:closure` test checks its dependency closure.

## Wiring

The packages here are unregistered, so a package that depends on a sibling lists it in `[deps]`
and in `[sources]`, with its relative path. MessagePassingRulesBase, MessagePassingRulesApproximations
and MessagePassingRulesTestUtils, registered, come from General; the test tooling is a test-only
dependency, in `[extras]`. `Pkg.test()` then works
from each package's own project, and so does its `make` target:

```bash
make test-standard test-delta \
     test-gaussian-coupling test-probit test-gcv test-autoregressive test-softdot \
     test-continuous-transition test-polya test-bifm test-flow test-discrete-transition
```

Each target takes `test_args` as the root's `make test` does. Items tagged `:slow` are skipped
unless `TEST_ALL=true`, which CI sets. `[sources]` needs Julia 1.11 or later; the packages are
developed and tested on Julia 1.13. No `Manifest.toml` under `lib/` is committed.
