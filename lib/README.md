# `lib/` — the new message-passing rule packages

The packages of the rule/node rewrite, created as stubs in Phase P. `MessagePassingRulesBase`
(Phase 3) and `MessagePassingRulesTestUtils` (Phase 4) are implemented, each with its own
test suite. `StandardMessagePassingRules` is being filled in Phase 4.5, starting with the
rules the engine slice needs, and `MessagePassingRulesApproximations` is still a
`Project.toml` plus an empty module.

Read `PLAN.md` for the design, `PHASES.md` for what is next, and `INVENTORY.md` for where
each of the 231 entities in ReactiveMP is destined to land.

| package | phase | role |
|---|---|---|
| `MessagePassingRulesBase` | 3 | macros, targets, algorithms, argument/annotation containers, context, registry, dependency language, `buffer_like` (not `Message`/`Marginal` — those stay in the engine) |
| `MessagePassingRulesTestUtils` | 4 | test tooling, consumed via `[extras]`: table tests, coverage, verification against the node definition, derivative checks, the migration checker |
| `StandardMessagePassingRules` | 5 | distributions, arithmetic, logic, mixtures |
| `MessagePassingRulesApproximations` | 6 | numerical utilities; **standalone, must not depend on the base** |

The domain-models package (`GCV`, `Probit`, `SoftDot`, `GaussianCoupling`) is not created
yet; its name is deliberately deferred to Phase 6, and `INVENTORY.md` records its
destination as the placeholder token `models`.

## Two constraints that are easy to erode

- `MessagePassingRulesBase` must **never** depend on `ExponentialFamily`. BayesBase exists
  to hold that machinery; if something is missing there, add it to BayesBase. Those
  additions must ship as non-breaking **1.x** releases, or `compat/v6-comparison` stops
  resolving and the Phase 4 migration checker goes with it.
- `MessagePassingRulesApproximations` must **never** depend on `MessagePassingRulesBase`.
  They are siblings: approximating an integral is numerics, choosing an update scheme is an
  algorithm.

## Wiring the inter-package dependencies

These packages are unregistered, and `[sources]` — the tidy way to point a `Project.toml` at
a sibling directory — requires Julia 1.11, while the floor is 1.10. So a package that depends
on another one here lists it in `[deps]` like any other dependency — and in `[sources]`,
which 1.11+ honours and 1.10 ignores — and the sibling is **developed into its environment
at test time**:

```julia
julia> using Pkg
julia> Pkg.activate("lib/MessagePassingRulesTestUtils")
julia> Pkg.develop(path = "lib/MessagePassingRulesBase")
julia> Pkg.test()
```

`make test-testutils` and `LibTests.yml` do exactly this. No `Manifest.toml` under `lib/` is
committed: each Julia version resolves for itself, which a manifest resolved on 1.10 could not
do for 1.11 and 1.12. Run the example from the repository root, since `Pkg.activate` changes
the active environment, not the working directory.

### A package that needs the test tooling

`MessagePassingRulesTestUtils` is a test-only dependency, and unregistered. On the 1.10
floor, developing it into a package's own project would make it a runtime dependency. So a
rule package such as `StandardMessagePassingRules` keeps its test dependencies in
`test/Project.toml`, with `[sources]` for 1.11+, and runs its suite from that environment
after developing the siblings into it. TestItemRunner finds the package from the test
file's location, not from the active project:

```bash
make test-standard     # does exactly this; LibTests.yml's `standard` job too
```

## Promotion to separate repositories

Planned for **Phase 6**, once Phase 3 freezes the base API and Phase 4.5 proves the engine
interface. Until then a change spanning two packages is one commit; afterwards it is two
pull requests and a version pin. That is the trade this layout is making deliberately.
