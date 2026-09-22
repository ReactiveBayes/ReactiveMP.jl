# ReactiveMP.jl — working notes

Keep this document alive. If you learn something about this repository that the next
person would want to know, edit it here rather than rediscovering it.

## What this package is

ReactiveMP.jl is a **reactive message passing engine** for Bayesian inference on factor
graphs. It is the computational core of the [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl)
ecosystem and is not usually used directly:

- **GraphPPL.jl** — model specification (`@model`), produces the factor graph
- **ReactiveMP.jl** — *this package*: message passing over that graph
- **RxInfer.jl** — the user-facing umbrella package
- **ExponentialFamily.jl / BayesBase.jl** — distributions and their algebra
- **Rocket.jl** — the reactive streams library the engine is built on

Inference works by propagating `Message`s along edges and computing `Marginal`s at
variables, using *update rules* defined per factor node and per outbound edge. Rules are
ordinary Julia functions dispatched on the node type, the target edge, and the types of the
incoming messages/marginals.

## Repository layout

```
src/
  ReactiveMP.jl        module root; include order matters (see below)
  rule.jl              @rule, @marginalrule, @call_rule, @test_rules, rule errors   (large)
  message.jl           Message, DeferredMessage, MessageObservable, MessageMapping
  marginal.jl          Marginal, MarginalObservable, MarginalMapping
  variable.jl          AbstractVariable API
  variables/           randomvar, constvar, datavar
  nodes/
    nodes.jl           node traits, FactorNode, activate!, the @node macro
    interfaces.jl      NodeInterface, IndexedNodeInterface, ManyOf
    clusters.jl        local marginals / factorisation clusters
    dependencies.jl    which messages+marginals a rule receives; stream wiring
    equality.jl        equality-chain optimisation for high-degree variables
    predefined/        ~45 node definitions (@node + @average_energy)
  rules/               ~171 files, one directory per node, one file per target edge
  approximations/      Unscented, Linearization, CVI, quadrature rules
  annotations/         per-message metadata (@logscale and friends)
  score/               free energy: AverageEnergy, DifferentialEntropy
ext/                   weakdep extensions (Optimisers, ExponentialFamilyProjection)
test/                  mirrors src/ exactly
```

Include order in `src/ReactiveMP.jl` is load-bearing: `nodes/equality.jl` must precede
`variables/`, and all `@node` definitions must precede all `@rule` definitions (the `@rule`
macro queries the node registry at *macro expansion time*).

## Running things

All via `make` (run `make help` for the list).

```bash
make test                                  # the fast subset: everything except `:slow`
make test-all                              # everything, which is what CI runs
make test test_args="rules:normal_mean_variance"   # one directory
make test test_args="rules:beta:out"               # one file
make test test_args="tag:rules"                    # by tag
make test test_args="name:NormalMixture"           # by test-item name
make test test_args="tag:rules name:Beta"          # combined
RUN_AQUA=false make test                   # skip the slow Aqua checks
make format                                # apply formatting
make check-format                          # verify only, no writes
make docs                                  # build documentation
make test-base                             # lib/MessagePassingRulesBase's own suite
```

`test_args` takes three kinds of entry, and they compose:

- **path** — `a:b` maps to `test/a/b`, matched with `occursin`, so prefixes work
- **`tag:<name>`** — only items carrying that tag
- **`name:<text>`** — only items whose name contains that text

Entries of the same kind are OR'ed; different kinds are AND'ed.

Tests are `@testitem` blocks (414 of them across ~231 files), each self-contained and
independently runnable. Naming convention is `"rules:<Node>:<edge>"` for rule tests.

**Every test item carries a tag.** The taxonomy is `:rules` (194), `:nodes` (83), `:engine`
(130 — everything that is not a rule or node test), plus `:alloc` on the six items that
assert allocation counts and `:quality` on the inventory gate. `:slow` exists and is
**currently unused**: nothing has been measured as slow yet, so nothing claims to be. When
items are tagged `:slow` they disappear from `make test` and stay in `make test-all` and CI.

The fast default must never become a coverage reduction — CI sets `TEST_ALL=true`, so a
`:slow` tag changes what *you* run locally, never what CI runs.

Rule tests are table-driven via `@test_rules`, which is defined in `src/rule.jl` (not in
`test/`) and is unexported — tests do `import ReactiveMP: @test_rules`. `Test` must be
imported by the caller.

## Conventions

- **`CHANGELOG.md` must be updated** in every PR — CI enforces it.
- Formatting is **Runic**, checked in CI (`make check-format`). It is zero-config — there is
  no style file, and `.JuliaFormatter.toml` is gone — and deterministic: measured, it produces
  byte-identical output on Julia 1.10 and 1.13, which is what JuliaFormatter could not do. The
  version is still pinned via `scripts/Manifest.toml`, since Runic's own output may change
  between releases; use `make scripts_update` to bump it deliberately, and run `make format`
  over the repo in the same commit. `docs/` is excluded, as it was before.
- Julia compat floor is currently 1.10.

## Gotchas

- Interface names in `@node` **may not contain underscores** — `_` is the separator used to
  parse joint-marginal names like `q_y_x` back into a cluster.
- `Marginalisation` in every `@rule` signature is a dead dispatch axis; it is hardcoded
  everywhere and `MomentMatching` is never dispatched on.
- Aqua's `ambiguities` check is **deliberately disabled** in `test/runtests.jl` (322 pairs,
  revisited after the split — see `PHASES.md` § Phase 2). `piracies` is on, with two owners
  declared through `treat_as_own`, and `deps_compat` checks `[extras]` too.
- `lib/` holds the new packages. `MessagePassingRulesBase` has its own test suite
  (`make test-base`, same `test_args` syntax) and CI job (`LibTests.yml`); the other three
  are still empty stubs with no tests.
- `src/fixes.jl` holds deliberate hot-fixes for upstream packages; it is expected to be
  empty when everything upstream has released.

## Ongoing work — read before starting

This repository may carry in-progress design work in the root. Look for these, in order,
and read whichever exist before proposing changes:

1. **`PLAN.md`** — the agreed design. What was decided.
2. **`DISCUSSION.md`** — why it was decided, which alternatives were rejected, and a
   **Corrections** section listing ideas already tried and shot down. Read that section
   before proposing a design; it exists to stop good-sounding dead ends being re-proposed.
3. **`PHASES.md`** — current state: what is done, what is next, what is blocked.
4. **`INVENTORY.md`** — where every node, exported symbol and engine hook is going once
   the package split happens. Generated by `scripts/inventory.jl`: edit only the
   `destination` and `note` columns, then run `julia --project=. scripts/inventory.jl
   --check` to confirm nothing is left `undecided`.

**If none of these files exist, the repository has no unfinished business** and you can
treat `main` as the whole story.

When work is in progress, update `PHASES.md` **in the same commit as the change it
describes**. Never mark something done as a separate act — status claimed without a diff
alongside it is how a tracking file starts lying.
