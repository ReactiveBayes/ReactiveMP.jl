# `lib/` — the new message-passing rule packages

Scaffolding for the rule/node rewrite. **Nothing here is implemented yet**; each package is
a `Project.toml` plus an empty module, created in Phase P so that Phase 3 onwards starts
with a working layout rather than inventing one under time pressure.

Read `PLAN.md` for the design, `PHASES.md` for what is next, and `INVENTORY.md` for where
each of the 230 entities in ReactiveMP is destined to land.

| package | phase | role |
|---|---|---|
| `MessagePassingRulesBase` | 3 | macros, `Message`/`Marginal`, targets, algorithms, context, registry, dependency language, `buffer_like` |
| `MessagePassingRulesTestUtils` | 4 | test tooling, consumed via `[extras]` |
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

These packages are unregistered, and `[sources]` — the tidy way to point a `Project.toml`
at a sibling directory — requires Julia 1.11, while the floor is 1.10. So a package that
depends on another here is wired with an explicit dev-link, and the resulting `Manifest.toml`
is committed so everyone resolves the same way:

```julia
julia> using Pkg
julia> Pkg.activate("lib/StandardMessagePassingRules")
julia> Pkg.develop(path = "../MessagePassingRulesBase")
```

Until that happens, the cross-package `[deps]` entries are left out and noted in each
`Project.toml`, so that every package here instantiates on its own.

## Promotion to separate repositories

Planned for **Phase 6**, once Phase 3 freezes the base API and Phase 4.5 proves the engine
interface. Until then a change spanning two packages is one commit; afterwards it is two
pull requests and a version pin. That is the trade this layout is making deliberately.
