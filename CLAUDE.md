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
make test                                  # everything (slow; Aqua enabled by default)
make test test_args="rules:normal_mean_variance"   # one directory
make test test_args="rules:beta:out"               # one file
RUN_AQUA=false make test                   # skip the slow Aqua checks
make format                                # apply formatting
make check-format                          # verify only, no writes
make docs                                  # build documentation
```

`test_args` entries are **path filters**: `a:b` maps to `test/a/b` and matches by
`occursin`, so prefixes work. There is currently **no way to filter by test name** — the
runner is `TestItemRunner` and `runtests.jl` only filters on filename.

Tests are `@testitem` blocks (~413 of them across ~231 files), each self-contained and
independently runnable. Naming convention is `"rules:<Node>:<edge>"` for rule tests.

Rule tests are table-driven via `@test_rules`, which is defined in `src/rule.jl` (not in
`test/`) and is unexported — tests do `import ReactiveMP: @test_rules`. `Test` must be
imported by the caller.

## Conventions

- **`CHANGELOG.md` must be updated** in every PR — CI enforces it.
- Formatting is checked in CI (`make check-format`). The formatter version is pinned via
  `scripts/Manifest.toml` on purpose; use `make scripts_update` to bump it deliberately
  rather than letting it drift.
- Julia compat floor is currently 1.10.

## Gotchas

- Interface names in `@node` **may not contain underscores** — `_` is the separator used to
  parse joint-marginal names like `q_y_x` back into a cluster.
- `Marginalisation` in every `@rule` signature is a dead dispatch axis; it is hardcoded
  everywhere and `MomentMatching` is never dispatched on.
- Aqua's `ambiguities` and `piracies` checks are currently **disabled** in `test/runtests.jl`.
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
