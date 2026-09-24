# `lib/` — the new message-passing rule packages

The packages of the rule/node rewrite. The first four were created as stubs in Phase P:
`MessagePassingRulesBase` (Phase 3) and `MessagePassingRulesTestUtils` (Phase 4) are
implemented, each with its own test suite, and `StandardMessagePassingRules` and
`MessagePassingRulesApproximations` were filled in Phase 4.5 with the rules and the numerics
the engine slice needs. `DeltaMessagePassingRules` was created in Phase 4.5 case (d). Phase 5
completed `StandardMessagePassingRules`; Phase 6 completes the others and adds a package per
non-standard node.

Read `PLAN.md` for the design, `PHASES.md` for what is next, and `INVENTORY.md` for where
each of the 231 entities in ReactiveMP is destined to land.

| package | phase | role |
|---|---|---|
| `MessagePassingRulesBase` | 3 | macros, targets, algorithms, argument/annotation containers, context, registry, dependency language, `buffer_like` (not `Message`/`Marginal` — those stay in the engine) |
| `MessagePassingRulesTestUtils` | 4 | test tooling, consumed via `[extras]` and `[sources]`: table tests, coverage, verification against the node definition, derivative checks, the migration checker |
| `StandardMessagePassingRules` | 4.5 (the slice's six nodes), then 5 (done) | distributions, arithmetic, logic, mixtures |
| `MessagePassingRulesApproximations` | 4.5 (`Unscented`, `smoothRTS`), then 6 (`Linearization`, Gauss–Hermite cubature) | numerical utilities over means and covariances; **standalone, must not depend on the base**, nor on a distribution package: `LinearAlgebra`, `FastCholesky`, `ForwardDiff` and `FastGaussQuadrature` |
| `DeltaMessagePassingRules` | 4.5 (`Unscented`), then 6 (`Linearization`, `CVIProjection`) | the Delta node (`INVENTORY.md`'s `node:Delta`, created early in case (d)): `DeltaFn{F}`, its algorithm `DeltaApproximation(; method, inverse)`, its dependencies and rules. The engine owns the node's function and static inputs |
| `GaussianCouplingMessagePassingRules` | 6 (step 3) | the GaussianCoupling node, the edge potential of Gaussian belief propagation |
| `ProbitMessagePassingRules` | 6 (step 3) | the Probit node and its algorithm `ProbitEP`, expectation propagation |
| `GCVMessagePassingRules` | 6 (step 3) | the GCV node, its `ExponentialLinearQuadratic`, and the normal nodes' rules for it |
| `AutoregressiveMessagePassingRules` | 6 (step 4) | the AR and ConjugateAR nodes, their algorithm `ARVMP`, and the companion matrix and standard basis vector their rules use |
| `SoftDotMessagePassingRules` | 6 (step 4) | the SoftDot node, with its own copy of the little AR algebra it needs rather than a dependency on AR's package |
| `ContinuousTransitionMessagePassingRules` | 6 (step 5) | the ContinuousTransition node and its algorithm `CTVMP(f)`; its rule towards `a` extends the default scheme with `q(a)` |
| `PolyaMessagePassingRules` | 6 (step 6) | BinomialPolya and MultinomialPolya, Pólya-Gamma augmented; **GPL-3** through PolyaGammaHybridSamplers, the only package that carries it |

There is no shared domain-models package: `GCV`, `Probit`, `SoftDot` and `GaussianCoupling`
each get their own node package in Phase 6, like the other non-standard nodes
(`DISCUSSION.md` §3.30), and `INVENTORY.md` records each as `node:<Name>`.

## Two constraints that are easy to erode

- `MessagePassingRulesBase` must **never** depend on `ExponentialFamily`. BayesBase exists
  to hold that machinery; if something is missing there, add it to BayesBase. Those
  additions must ship as non-breaking **1.x** releases, or `compat/v6-comparison` stops
  resolving and the Phase 4 migration checker goes with it.
- `MessagePassingRulesApproximations` must **never** depend on `MessagePassingRulesBase`.
  They are siblings: approximating an integral is numerics, choosing which rules run is an
  algorithm.

## Wiring the inter-package dependencies

These packages are unregistered, so a package that depends on a sibling lists it in `[deps]`
like any other dependency, and in `[sources]` with its relative path. A test-only sibling,
such as `MessagePassingRulesTestUtils` for a rule package, goes in `[extras]` and `[sources]`
the same way. Plain `Pkg.test()` then works, and so does every `make test-*` target:

```bash
make test-base test-testutils test-standard test-approximations test-delta test-gaussian-coupling test-probit test-gcv test-autoregressive test-softdot test-continuous-transition test-polya
```

Work targets **Julia 1.13**, where `[sources]` is honoured (1.11+). On the old 1.10 floor it was
ignored, which cost a develop-at-test-time step for every sibling and a separate
`test/Project.toml` for a test-only one; both are gone since Phase 4.5 step 4, and 1.10 support
is reconsidered at registration (`DISCUSSION.md` §3.22). No `Manifest.toml` under `lib/` is
committed; the local ones are gitignored.

## Promotion to separate repositories

Planned for **Phase 8**, with registration (`DISCUSSION.md` §3.40); it was Phase 6 until the
Phase 6 brief, since Phase 3 froze the base API and Phase 4.5 proved the engine interface, but
the node packages are wired with `[sources]` and run no CI until they are registered. Until then a change spanning two packages is one commit; afterwards it is two
pull requests and a version pin. That is the trade this layout is making deliberately.
