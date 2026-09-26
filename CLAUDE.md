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
  scratch.jl           ScratchSlot: the rule scratch each message and marginal mapping keeps
  diagnostics.jl       EngineDiagnostics: the opt-in purity, in-place and checked-buffer audits
  rule_arguments.jl    RuleArgs/RuleAnnotations built from the latest messages and marginals
  context.jl           node_context: the RuleContext a node's rules read, the engine's services and the model's
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
  logscale.jl          log scales: products, constants, initial messages
  annotations/         per-message metadata (input arguments)
  callbacks.jl         rule-call and other engine events
  postprocessors.jl    stream postprocessors (with postprocessors/)
  constraints/         form constraints
  helpers/             small internal utilities
  score/               node scores, variable entropies, bethe_free_energy
  fixes.jl             upstream hot-fixes; empty now
lib/                   the new packages: the rule system, its test tooling, rules, numerics
compat/v6-comparison/  ReactiveMP 6.5.0 + RxInfer 5.5.2: the oracle, comparisons, engine fixtures
compat/rxinfer-examples/ five RxInferExamples models on v6 and on RxInfer's v7 branch
investigations/        performance investigations kept for the end-of-refactor pass; never loaded
test/                  mostly mirrors src/; test/engine/ runs whole graphs against closed forms and invariants
```

Include order in `src/ReactiveMP.jl` is load-bearing: `nodes/equality.jl` must precede
`variables/`. Nodes and rules are defined with `MessagePassingRulesBase`'s macros
(`@define_factor_node`, `@define_message_update_rule`, …), and a rule that omits `algorithm`
must load after its node's declaration.

## Running things

All via `make` (run `make help` for the list).

```bash
make test                                  # the root suite, except items tagged `:slow`
make test-all                              # the root suite, `:slow` included (as CI runs it)
make test test_args="nodes"                        # one directory
make test test_args="engine:variational"           # one file
make test test_args="tag:engine"                   # by tag
make test test_args="name:MessageMapping"          # by test-item name
make test test_args="tag:nodes name:factornode"    # combined
RUN_AQUA=false make test                   # skip the slow Aqua checks
make format                                # apply formatting
make check-format                          # verify only, no writes
make docs                                  # build the documentation, running its doctests
make test-base                             # lib/MessagePassingRulesBase's own suite
make test-testutils                        # lib/MessagePassingRulesTestUtils, against the local base
make test-standard                         # lib/StandardMessagePassingRules
make test-approximations                   # lib/MessagePassingRulesApproximations, which depends on no sibling
make test-delta                            # lib/DeltaMessagePassingRules
make test-gaussian-coupling                # lib/GaussianCouplingMessagePassingRules
make test-probit                           # lib/ProbitMessagePassingRules
make test-gcv                              # lib/GCVMessagePassingRules
make test-autoregressive                   # lib/AutoregressiveMessagePassingRules: AR, ConjugateAR
make test-softdot                          # lib/SoftDotMessagePassingRules
make test-continuous-transition            # lib/ContinuousTransitionMessagePassingRules
make test-polya                            # lib/PolyaMessagePassingRules (GPL-3)
make test-bifm                             # lib/BIFMMessagePassingRules
make test-flow                             # lib/FlowMessagePassingRules
make test-discrete-transition              # lib/DiscreteTransitionMessagePassingRules
# the v6 oracle environment: comparisons and engine fixtures
for s in check compare_standard compare_approximations compare_delta compare_gaussian_coupling compare_probit compare_gcv compare_autoregressive compare_softdot compare_continuous_transition compare_polya compare_bifm compare_flow compare_discrete_transition; do julia --project=compat/v6-comparison compat/v6-comparison/$s.jl; done
julia --project=compat/v6-comparison compat/v6-comparison/record_engine_fixtures.jl --check
```

Work targets **Julia 1.13**, and **no CI runs** until a PR is opened: every check above is run
locally. The workflows under `.github/` run these same checks, on 1.13 (the format check on
1.10, where Runic's output is byte-identical).

`test_args` takes three kinds of entry, and they compose:

- **path** — `a:b` maps to `test/a/b`, matched with `occursin`, so prefixes work
- **`tag:<name>`** — only items carrying that tag
- **`name:<text>`** — only items whose name contains that text

Entries of the same kind are OR'ed; different kinds are AND'ed.

Tests are `@testitem` blocks (174 of them across 27 files), each self-contained and
independently runnable. The root suite skips `lib/` and `compat/`, which
TestItemRunner would otherwise scan. `@testmodule` names are global across the whole
directory, `lib/` included, so a new one must not reuse a name from a lib suite.

**Every test item carries a tag.** The taxonomy is `:nodes` (29) and `:engine` (143 —
everything except the node tests and the quality items), plus `:alloc` on the two items that
assert allocation counts and `:quality` on the inventory gate and the engine's doctests. Rules
are tested in the lib suites. `:slow` exists and is **unused in `test/`**: nothing there has been
measured as slow, so nothing claims to be.
The lib suites honour it the same way: `registry:lifecycle` in `MessagePassingRulesBase` is
`:slow`, so `make test-base` skips it unless you set `TEST_ALL=true`
(`TEST_ALL=true make test-base`). When items are tagged `:slow` they disappear from `make test`
and stay in `make test-all`.

The fast default must never become a coverage reduction — the CI workflows set `TEST_ALL=true`,
so a `:slow` tag changes what *you* run locally, never what CI runs. `ci.yml` runs the root
suite and the docs, `LibTests.yml` a job per `make test-<package>` target and one for the v6
comparison, the fixtures and the inventory; until the first PR "CI" means these checks run
locally.

Rule tests live with the rules, in the lib packages, and are table-driven via
`MessagePassingRulesTestUtils` (`@test_message_update_rule`). The engine's own tests declare
toy nodes and rules with the base package's macros. `test/engine/harness.jl` builds a graph the
way RxInfer does and returns its posteriors and free energies; the engine tests check them
against closed forms computed in the test (exact models), coordinate ascent or convergence
(variational ones) and quadrature or Monte Carlo (approximations). Two known rule
discrepancies are `@test_broken` there: AR's mean-field rule towards `γ` and CT's rule towards
`y` from `m[:x]`.

## Conventions

- **`CHANGELOG.md` must be updated** in every change (CI enforces it on a PR).
- Formatting is **Runic** (`make check-format`; CI checks it on a PR). It is zero-config — there is
  no style file — and deterministic: measured, it produces byte-identical output on Julia 1.10
  and 1.13. The version is still pinned via `scripts/Manifest.toml`, since Runic's own output may change
  between releases; use `make scripts_update` to bump it deliberately, and run `make format`
  over the repo in the same commit. `docs/` is excluded.
- Julia: work targets **1.13 only**, and siblings are wired with `[sources]`, test-only ones via
  `[extras]` too; the comparison environment is resolved on 1.13 as well. The lowest supported
  version is decided when the packages are registered.

## Gotchas

- `factornode` takes interfaces as `(name, variable)` or `((group, k), variable)` and a
  factorisation as tuples of those keys, `((:out, :μ), (:v,))` — never positions. It puts
  everything in declaration order. A joint local marginal is keyed by its member tuple,
  `(:out, :μ)`, so interface names may contain underscores.
- Rule lookup is the base package's method table (`find_message_rule` and friends), global
  across every loaded package. The per-module `__message_passing_registry__` is introspection
  only, per module because of precompilation; the engine never reads it.
- `activate!` wires a node's declared dependencies (`dependencies_spec`) or the default scheme,
  groups included, and subscribes to a target's inputs **in declaration order**, which in VMP
  is the update schedule. A joint may hold some members of a group,
  keyed with them, `(:out, (:T, 1))`. A declaration may write `default` among a target's inputs, `:a => (default, q[:a])`:
  the default scheme's inputs plus the listed ones, placed in interface order. A
  deterministic node's clusters are always `out` and the joint over its inputs,
  and a `static_inputs = :fold` node needs `factornode(…; nodefn = f)`.
- Aqua's checks run in full in `test/runtests.jl`, `ambiguities` included; `deps_compat` checks
  `[extras]` too.
- A rule reads its services from `ctx`, a `RuleContext` wrapping a `NamedTuple`: the engine
  supplies `node`, `rng` and `matrix_correction` (`node_context`), and the activation
  option `context`, any `NamedTuple`, is merged over them. A name nobody supplies reads as
  `nothing` in an interactive call; the engine refuses a rule that declares one
  (`check_services`, as it resolves the rule). The free energy's average energies see only the
  engine's services, not the `context` option.
- The activation option `rulefallback` (e.g. `NodeFunctionRuleFallback()`) gives a message only
  where no rule matches; an exception inside a rule always propagates.
- `lib/` holds the rule packages, each with its own suite and the same `test_args` syntax:
  `MessagePassingRulesBase` (`make test-base`), `MessagePassingRulesTestUtils`
  (`make test-testutils`), `StandardMessagePassingRules` (`make test-standard`; every standard node: the distributions, arithmetic, logic and the mixtures), `MessagePassingRulesApproximations` (`make test-approximations`;
  `Unscented`, `Linearization`, Gauss–Hermite cubature and `smoothRTS`, pure numerics) and `DeltaMessagePassingRules`
  (`make test-delta`; the Delta node, its algorithm `DeltaApproximation`, its Unscented and
  Linearization rules, and `CVIProjection`, whose rules are an extension on
  ExponentialFamilyProjection) and the node packages, one per node, on Delta's template:
  `GaussianCouplingMessagePassingRules` (`make test-gaussian-coupling`),
  `ProbitMessagePassingRules` (`make test-probit`), `GCVMessagePassingRules` (`make test-gcv`),
  `AutoregressiveMessagePassingRules` (`make test-autoregressive`; AR and ConjugateAR under
  `ARVMP`, which the model must give), `SoftDotMessagePassingRules` (`make test-softdot`,
  independent of AR's package), `ContinuousTransitionMessagePassingRules`
  (`make test-continuous-transition`; `CTVMP(f)`, its rule towards `a` reading `q(a)` beside the
  default scheme's inputs), `PolyaMessagePassingRules` (`make test-polya`; BinomialPolya and
  MultinomialPolya, GPL-3 through PolyaGammaHybridSamplers, the only package that is),
  `BIFMMessagePassingRules` (`make test-bifm`; BIFM, stateless, and BIFMHelper; no free energy)
  `FlowMessagePassingRules` (`make test-flow`; Flow, its flow models and `PermutationMatrix`) and
  `DiscreteTransitionMessagePassingRules` (`make test-discrete-transition`; a tensor node, its rules
  `default` ones walking `rule_inputs`, its `T` group possibly empty). Siblings are wired with `[deps]` and `[sources]`. No Manifest under `lib/` is committed; the local ones are gitignored.
- The Standard and Delta suites end with a **rule-coverage gate**: after an unfiltered run
  (no `test_args`, and `TEST_ALL=true` if anything is `:slow`), `check_rule_coverage` must
  find every rule selected by some test. A table case, a verification, a derivative check or
  a direct `call_*` counts; a rule reached only through a graph does not. A new rule needs a test
  in its package, or `make test-standard` fails.
- A node's `algorithm = T` with a **parametric** `T` binds its inline `dependencies`, and every rule
  that omits `algorithm`, to the type of its default instance, `T{Nothing}` say, not to `T`. A
  node whose algorithm has variants declares them against `T` itself, with `@define_dependencies`
  and `algorithm = T` on the rules, as `BinomialPolyaApproximation` does.
- `@define_factor_node` can declare what a node requires of the graph: `matched_groups =
  [(:m, :p)]`, `min_group_length = 2`, `factorisation = :meanfield`. `factornode` checks them;
  the two mixtures declare all three.
- The v6 → v7 migration guide is the docs page `docs/src/migration-guides/v6-to-v7.md`; there is
  no `MIGRATION.md`.
- A **log scale** is part of a message (`Message{D, L}`, `Marginal{D, L}`), not an annotation:
  the scalar with `message = exp(logscale) · distribution`. A rule declares it with the
  `logscale` keyword (a constant, a function of `args`, or `from_body` with the body returning
  `with_logscale`); one that omits it gives an `UndefinedLogScale` naming it, which propagates
  and errors only where a number is needed (`require_logscale`). The engine tracks them only
  with the activation option `logscales = true`; otherwise messages carry `nothing`. A rule
  reading its inputs' ones declares `reads_logscale = true` and reads `args.logscale.m[:x]`.
- Every public call of a rule (`call_*`, `@call_*`, `message_passing_*`) returns a
  `RuleResult`; `getresult` is the value, `getlogscale` its log scale. The engine uses
  `execute_rule`/`execute_rule_with_logscale` and never builds one; inlined,
  `getresult(message_passing_rule(...))` still allocates nothing (the routing gates).
- The v6 code is only in the 6.5.0 release and in git. The inventory gate runs in
  `compat/v6-comparison`, since only v6.5.0 still has everything it enumerates.
- `visualize_spec` is a deliberate entry point for visualisation backends (extensions), none
  written yet: comprehensive visualisations of nodes, dependencies and rules, rendered in the
  documentation. Not dead code.
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

The current work is the rule/node rewrite; Phases 4.5 and 5 are closed, with the post-close
review's findings resolved; Phase 6 (the node packages) is closed: every node is in its own package and `legacy/` is gone. Phase 7 is closed: RxInfer is adapted on its branch `refactor/reactivemp-v7` (pushed, no PR, run by `IntegrationTest.yml`), the diagnostics are activation options (`DISCUSSION.md` §3.46). **Phase C, the cleanup, is in progress**: it has also brought back rule fallbacks and opened the rule context (§3.49), made log scales part of the message with `RuleResult` (§3.50, superseding §3.48's "as v6 has them") and given `RuleResult` its display (§3.51). The engine checks a rule's declared services as it resolves it; before the release, the remaining 27 log scales and the performance pass (`PHASES.md`, *Next action* and the not-done table).
From Phase 4.5 on, the engine in `src/` is **refactored in place**, not bridged. Its reactive
machinery is kept, and rule lookup and invocation plus node and rule definition and creation
are replaced. Step 4 was a **clean cut**: the v6 rule system and every unported node moved to
`legacy/v6/`, after their behaviour was recorded as fixtures in `compat/v6-comparison`; Phase 5
step 9 then deleted the v6 engine files from there, and Phase 6 ported the rest and deleted the
directory. Breaking downstream
packages before the release is accepted.

Remarks in code and tests that only record the rewrite's history (phases, steps, cases, what
v6 did, pointers into these documents) are already out of `src/` and `lib/`; do not add new
ones. What remains (the inventory, `compat/`, these documents, the CHANGELOG's step-by-step
entries) waits for the release, and **Phase C** removes it then (`PHASES.md` § Phase C).

When work is in progress, update `PHASES.md` **in the same commit as the change it
describes**. Never mark something done as a separate act — status claimed without a diff
alongside it is how a tracking file starts lying.
