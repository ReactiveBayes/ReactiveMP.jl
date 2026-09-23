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
  message.jl           Message, DeferredMessage, MessageObservable, MessageMapping
  marginal.jl          Marginal, MarginalObservable, MarginalMapping
  rule_arguments.jl    RuleArgs/RuleAnnotations built from the latest messages and marginals
  variable.jl          AbstractVariable API
  variables/           randomvar, constvar, datavar
  nodes/
    nodes.jl           FactorNode, factornode (from a @define_factor_node declaration), activate!
    interfaces.jl      NodeInterface, IndexedNodeInterface
    clusters.jl        local marginals / factorisation clusters, keyed :μ or (:out, :μ)
    dependencies.jl    declared and default dependencies; group inputs; stream wiring
    static_inputs.jl   StaticFold, getnodefn on FactorNode, with_statics
    equality.jl        equality-chain optimisation for high-degree variables
  annotations.jl       AnnotationDict, the per-message annotation store
  annotations/         per-message metadata (log scale, input arguments)
  callbacks.jl         rule-call and other engine events
  postprocessors.jl    stream postprocessors (with postprocessors/)
  constraints/         form constraints
  helpers/             small internal utilities
  score/               node scores, variable entropies, bethe_free_energy
  fixes.jl             upstream hot-fixes; empty now
lib/                   the new packages: the rule system, its test tooling, rules, numerics
compat/v6-comparison/  ReactiveMP 6.5.0 + RxInfer 5.5.2: the oracle, comparisons, engine fixtures
legacy/v6/             the v6 rule system and unported nodes, for reference; never loaded
test/                  mostly mirrors src/; test/engine/ runs whole graphs against the v6 fixtures
```

Include order in `src/ReactiveMP.jl` is load-bearing: `nodes/equality.jl` must precede
`variables/`. Nodes and rules are defined with `MessagePassingRulesBase`'s macros
(`@define_factor_node`, `@define_message_update_rule`, …), and a rule that omits `algorithm`
must load after its node's declaration.

## Running things

All via `make` (run `make help` for the list).

```bash
make test                                  # the fast subset: everything except `:slow`
make test-all                              # everything, including `:slow`
make test test_args="nodes"                        # one directory
make test test_args="engine:fixtures"              # one file
make test test_args="tag:engine"                   # by tag
make test test_args="name:MessageMapping"          # by test-item name
make test test_args="tag:nodes name:factornode"    # combined
RUN_AQUA=false make test                   # skip the slow Aqua checks
make format                                # apply formatting
make check-format                          # verify only, no writes
make docs                                  # refuses: docs/ describes v6 and is being rewritten
make test-base                             # lib/MessagePassingRulesBase's own suite
make test-testutils                        # lib/MessagePassingRulesTestUtils, against the local base
make test-standard                         # lib/StandardMessagePassingRules
make test-approximations                   # lib/MessagePassingRulesApproximations, which depends on no sibling
make test-delta                            # lib/DeltaMessagePassingRules
# the v6 oracle environment: comparisons and engine fixtures
for s in check compare_standard compare_approximations; do julia --project=compat/v6-comparison compat/v6-comparison/$s.jl; done
julia --project=compat/v6-comparison compat/v6-comparison/record_engine_fixtures.jl --check
```

Work targets **Julia 1.13** for now, and **no CI runs** until a PR is opened: every check above
is run locally. The workflow files under `.github/` are left as they are until registration.

`test_args` takes three kinds of entry, and they compose:

- **path** — `a:b` maps to `test/a/b`, matched with `occursin`, so prefixes work
- **`tag:<name>`** — only items carrying that tag
- **`name:<text>`** — only items whose name contains that text

Entries of the same kind are OR'ed; different kinds are AND'ed.

Tests are `@testitem` blocks (130 of them across 21 files), each self-contained and
independently runnable. The root suite skips `legacy/`, `lib/` and `compat/`, which
TestItemRunner would otherwise scan. `@testmodule` names are global across the whole
directory, `lib/` included, so a new one must not reuse a name from a lib suite.

**Every test item carries a tag.** The taxonomy is `:nodes` (21) and `:engine` (108 —
everything except the node tests and the inventory gate), plus `:alloc` on the two items that
assert allocation counts and `:quality` on the inventory gate. `:rules` went with the v6 rule
tests; rules are tested in the lib suites now. `:slow` exists and is **unused in `test/`**: nothing there has been measured as slow yet, so nothing claims to be.
The lib suites honour it the same way: `registry:lifecycle` in `MessagePassingRulesBase` is
`:slow`, so `make test-base` skips it unless you set `TEST_ALL=true`
(`TEST_ALL=true make test-base`). When items are tagged `:slow` they disappear from `make test`
and stay in `make test-all`.

The fast default must never become a coverage reduction — the CI workflows set `TEST_ALL=true`,
so a `:slow` tag changes what *you* run locally, never what CI runs. The workflows under
`.github/` still describe the 1.10 matrix and the pre-step-4 layout, and are brought up to date
before the first PR (`PHASES.md` § Phase 7); until then, "CI" means these checks run locally.

Rule tests live with the rules, in the lib packages, and are table-driven via
`MessagePassingRulesTestUtils` (`@test_message_update_rule`). The engine's own tests declare
toy nodes and rules with the base package's macros. `test/engine/harness.jl` builds a graph the
way RxInfer does and records an `EngineTrajectory`, to compare with the v6 fixtures.

## Conventions

- **`CHANGELOG.md` must be updated** in every change (CI enforces it on a PR).
- Formatting is **Runic** (`make check-format`; CI checks it on a PR). It is zero-config — there is
  no style file, and `.JuliaFormatter.toml` is gone — and deterministic: measured, it produces
  byte-identical output on Julia 1.10 and 1.13, which is what JuliaFormatter could not do. The
  version is still pinned via `scripts/Manifest.toml`, since Runic's own output may change
  between releases; use `make scripts_update` to bump it deliberately, and run `make format`
  over the repo in the same commit. `docs/` is excluded, as it was before.
- Julia: work targets **1.13 only**, and siblings are wired with `[sources]`, test-only ones via
  `[extras]` too; the comparison environment is resolved on 1.13 as well. The 1.10 floor and
  its old workarounds are reconsidered when the packages are registered (`DISCUSSION.md` §3.22).

## Gotchas

- `factornode` takes interfaces as `(name, variable)` or `((group, k), variable)` and a
  factorisation as tuples of those keys, `((:out, :μ), (:v,))` — never positions. It puts
  everything in declaration order. A joint local marginal is keyed by its member tuple,
  `(:out, :μ)`, so interface names may contain underscores.
- Rule lookup is the base package's method table (`find_message_rule` and friends), global
  across every loaded package. The per-module `__message_passing_registry__` is introspection
  only, per module because of precompilation; the engine never reads it (`DISCUSSION.md` §3.23).
- `activate!` wires a node's declared dependencies (`dependencies_spec`) or the default scheme,
  groups included, and subscribes to a target's inputs **in declaration order**, which in VMP
  is the update schedule (`DISCUSSION.md` §3.24). It refuses a joint holding only some members
  of a group. A deterministic node's clusters are always `out` and the joint over its inputs,
  and a `static_inputs = :fold` node needs `factornode(…; nodefn = f)` (§3.25).
- Aqua's `ambiguities` check is **deliberately disabled** in `test/runtests.jl` (it was 322
  pairs on `main`, most in code now in `legacy/`; to be re-measured). `piracies` is on, and
  `deps_compat` checks `[extras]` too.
- `lib/` holds the new packages, each with its own suite and the same `test_args` syntax:
  `MessagePassingRulesBase` (`make test-base`), `MessagePassingRulesTestUtils`
  (`make test-testutils`), `StandardMessagePassingRules` (`make test-standard`; the slice's
  six nodes, the univariate distributions, the logic nodes, the multivariate normals, the matrix and Wishart nodes, the arithmetic nodes and the mixtures so far), `MessagePassingRulesApproximations` (`make test-approximations`;
  `Unscented` and `smoothRTS`, pure numerics) and `DeltaMessagePassingRules`
  (`make test-delta`; the Delta node, its algorithm `DeltaApproximation` and its Unscented
  rules). Siblings are wired with `[deps]` and `[sources]`. No Manifest under `lib/` is committed; the local ones are gitignored.
- `legacy/v6/` holds the v6 rule system and every node not yet ported: never loaded, never
  tested, kept as the reference Phase 5 ports from. The inventory gate runs in
  `compat/v6-comparison`, since only v6.5.0 still has everything it enumerates.
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
   `destination` and `note` columns, then run `julia --project=compat/v6-comparison
   scripts/inventory.jl --check` to confirm nothing is left `undecided`.

**If none of these files exist, the repository has no unfinished business** and you can
treat `main` as the whole story.

The current work is the rule/node rewrite; Phase 4.5 is closed, and Phase 5 is in progress (steps 1–8
done). From
Phase 4.5 on, the engine in `src/` is
**refactored in place**, not bridged. Its reactive machinery is kept, and rule lookup and
invocation plus node and rule definition and creation are replaced. Step 4 is a **clean
cut**: the v6 rule system and every unported node move to `legacy/v6/` (moved, not deleted),
after their behaviour was recorded as fixtures in `compat/v6-comparison`. Breaking downstream
packages before the release is accepted.

Remarks in code and tests that only record the rewrite's history (phases, steps, cases, what
v6 did, pointers into these documents) are allowed while it is in progress, and **Phase C**
removes them before the release, with these documents (`PHASES.md` § Phase C).

When work is in progress, update `PHASES.md` **in the same commit as the change it
describes**. Never mark something done as a separate act — status claimed without a diff
alongside it is how a tracking file starts lying.
