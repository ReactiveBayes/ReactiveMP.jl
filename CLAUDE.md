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
  scratch.jl           ScratchSlot: a rule's scratch kept by its mapping, made at the first call of a rule with scratch
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
ext/                   ReactiveMPReactantExt: graph building runs untraced under Reactant (see Gotchas)
test/                  mostly mirrors src/; test/engine/ runs whole graphs against closed forms and invariants;
                       test/reactant/ has its own environment, for the Reactant extension
docs/                  the engine's documentation site; each package's is in its repository
scripts/               the formatter (Runic, pinned); no benchmarks, which live in RxInferBenchmarks.jl
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
make docs                                  # ReactiveMP's site; it links to the rule packages' published sites
make test-reactant                         # compile an unrolled model with Reactant, from test/reactant's environment
```

The packages support **Julia 1.10 and later**, and local work runs on 1.13. The workflows under
`.github/` run the root suite on 1.10, 1.11, 1.12 and 1.13, all gating, and the docs on 1.13 (the
format check on 1.10, where Runic's output is byte-identical); `IntegrationTest.yml` runs RxInfer's
suite against this checkout, and *Reactant check* the Reactant extension. Local checks need only
1.13; CI covers the others.

`test_args` takes three kinds of entry, and they compose:

- **path** — `a:b` maps to `test/a/b`, matched with `occursin`, so prefixes work
- **`tag:<name>`** — only items carrying that tag
- **`name:<text>`** — only items whose name contains that text

Entries of the same kind are OR'ed; different kinds are AND'ed.

Tests are `@testitem` blocks, each self-contained and independently runnable. `@testmodule` names are global across the whole directory.

**Every test item carries a tag.** The taxonomy is `:nodes` (the node tests) and `:engine`
(everything else but the quality items), plus `:alloc` on the items that assert allocation
counts and `:quality` on the engine's doctests. Rules
are tested in the rule packages' suites. `:slow` exists and is **unused in `test/`**: nothing there has been
measured as slow, so nothing claims to be.
The rule packages' suites honour it the same way (Delta's CVIProjection item is `:slow`), and a
filtered run skips the rule-coverage gate, so a package's gate runs only with `TEST_ALL=true`.
When items are tagged `:slow` they disappear from `make test` and stay in `make test-all`.

The fast default must never become a coverage reduction — the CI workflows set `TEST_ALL=true`,
so a `:slow` tag changes what *you* run locally, never what CI runs. `ci.yml` runs the root
suite and the docs on PRs and on pushes to `main`. The rule packages are tested in their own
repositories, on every pull request and weekly (`Weekly.yml`).

Rule tests live with the rules, in the rule packages, and are table-driven via
`MessagePassingRulesTestUtils` (`@test_message_update_rule`). The engine's own tests declare
toy nodes and rules with the base package's macros. `test/engine/harness.jl` builds a graph the
way RxInfer does and returns its posteriors and free energies; the engine tests check them
against closed forms computed in the test (exact models), coordinate ascent or convergence
(variational ones) and quadrature or Monte Carlo (approximations). Nothing there is
`@test_broken`: the two rule discrepancies they found, AR's mean-field rule towards `γ` and CT's
rule towards `y` from `m[:x]` (ReactiveMP.jl#681, and #682 for the latter, which asks Thijs van de
Laar to triple-check it), are corrected in the rule packages.

## Conventions

- **`CHANGELOG.md` must be updated** in every change (CI enforces it on a PR).
- Formatting is **Runic** (`make check-format`; CI checks it on a PR). It is zero-config — there is
  no style file — and deterministic: measured, it produces byte-identical output on Julia 1.10
  and 1.13. The version is still pinned via `scripts/Manifest.toml`, since Runic's own output may change
  between releases; use `make scripts_update` to bump it deliberately, and run `make format`
  over the repo in the same commit. `docs/` is excluded.
- **Documentation**: every package has its own site, `docs/` for the engine and each package's in
  its repository, and a `README.md`. A site documents its own modules only (`checkdocs = :all`,
  strict), and links into another's with `` [`name`](@extref Package.name) ``, by its published
  `https://reactivebayes.github.io/<Pkg>.jl/stable/objects.inv`, through `registered(name)` in `make.jl`, which waits 30 s
  where DocInventories waits 1 s: a first request to GitHub Pages took 3 s here and failed the
  build. A plain `@ref` to another package's name fails the build. Node pages follow one template (overview, definition, interfaces,
  algorithm, a `rule_coverage` table, example, limitations, API). The docstring style is on the
  contributing page (`docs/src/extra/contributing.md`, *Documentation*): signature first, then
  only the sections that carry something, every documented name linked, shared text as
  interpolated `const DOC_*` fragments, limitations stated plainly. A rule package's docs show
  rule calls, never models: no GraphPPL `@model`, RxInfer `infer` or `@algorithm`, only a link
  to them from its overview. A name the docs tell users to
  call is public: exported, or declared with `Compat.@compat public`, which also parses on 1.10.
  Pages follow the writing rules on the same page (*Pages*): written to "you", one idea per
  sentence, terms linked to MessagePassingRulesBase's glossary (`glossary-<term>` anchors), and
  **every Julia block runs** at build time (`@example`, `@setup`, `@repl`, `jldoctest`; only
  shell and `# v6` halves are plain). **A site loads only its own package's dependencies**, plus
  BayesBase, ExponentialFamily and Distributions: the engine's site declares its one example node
  in `docs/nodes.jl` (every page includes it) and loads no rule package, and the sites of the base
  package and TestUtils, in their own repositories, declare toy nodes on the page; the learning path with the standard nodes is RxInfer's.
  A spec, a rule result or a declaration left as an `@example`'s last value renders as a card with
  the node drawn (MessagePassingRulesBase's `svg.jl`).
- Julia: the packages support **1.10 and later** (`julia = "1.10"`, stdlib compat entries 1.10);
  local work runs on 1.13, and every sibling comes from General.

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
- **CI sets `JULIA_FASTCHOLESKY_THROW_ERROR_NON_SYMMETRIC=1`**, which makes FastCholesky throw
  on a matrix that is not symmetric within 1e-8 instead of warning; set it on local runs too
  (`JULIA_FASTCHOLESKY_THROW_ERROR_NON_SYMMETRIC=1 TEST_ALL=true make test` in a package). Without it,
  ContinuousTransition's suite passed locally while CI failed for a week (Float32 rounding).
- Aqua's checks run in full in `test/runtests.jl`, `ambiguities` included; `deps_compat` checks
  `[extras]` too.
- A rule reads its services from `ctx`, a `RuleContext` wrapping a `NamedTuple`: the engine
  supplies `node`, `rng` and `matrix_correction` (`node_context`), and the activation
  option `context`, any `NamedTuple`, is merged over them. A name nobody supplies reads as
  `nothing` in an interactive call; the engine refuses a rule that declares one
  (`check_services`, as it resolves the rule). The free energy's average energies run with the
  context, diagnostics and callbacks their node was activated with, which `activate!` keeps on the
  node; a variable keeps its marginal context's callbacks for its entropy term.
- A callback event is built only where `listens` says a handler wants it (`@invoke_callback`), so
  anything an event needs beyond what the computation has (the rule that ran, say) is computed
  inside the event's constructor call, never on the path itself. `test/engine/tracing_tests.jl`
  pins that a handler listening to nothing costs nothing.
- A rule checks what its inputs' types cannot say with `args_check` (returning `true`, or `false`
  or a string saying why), raising a `RuleInputError`; a failed check never falls through to
  another rule. A combination a node does not support at all is a rule of its own that throws.
- The activation option `rulefallback` (e.g. `NodeFunctionRuleFallback()`) gives a message only
  where no rule matches; an exception inside a rule always propagates.
- **The sibling packages are registered** and live in their own repositories,
  `ReactiveBayes/<Package>.jl`: MessagePassingRulesBase (the rule system), MessagePassingRulesTestUtils
  (testing rules), MessagePassingRulesApproximations (the numerics rules share), and the rule
  packages: StandardMessagePassingRules (every standard node: the distributions, arithmetic, logic,
  the mixtures and Probit), Delta (`DeltaApproximation`, Unscented, Linearization, and
  `CVIProjection` as an extension on ExponentialFamilyProjection), GaussianCoupling, Autoregressive
  (AR and ConjugateAR under `ARVMP`), SoftDot, ContinuousTransition (`CTVMP(f)`), Polya (GPL-3
  through PolyaGammaHybridSamplers, the only package that is), BIFM, Flow, DiscreteTransition and
  GCV. ReactiveMP and RxInfer take them from General (`[compat] "1"`), and the sites link to their
  published inventories. A change to one goes to its repository; each has a `Makefile` (`make
  test`, `make docs`) and a weekly CI run (`Weekly.yml`), and a new one starts from
  `ReactiveBayes/MessagePassingRulesTemplate.jl`. Releasing is described on the contributing page
  (`docs/src/extra/contributing.md`, *Releasing*).
- Every rule package's suite ends with a
  **rule-coverage gate**: after an unfiltered run
  (no `test_args`, and `TEST_ALL=true` if anything is `:slow`), `check_rule_coverage` must
  find every rule selected by some test. A table case, a verification, a derivative check or
  a direct `call_*` counts; a rule reached only through a graph does not. A new rule needs a test
  in its package, or its suite fails.
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
  with the activation option `logscales = true`; otherwise messages carry `nothing`, and data and
  constants zero. A rule
  reading its inputs' ones declares `reads_logscale = true` and reads `args.logscale.m[:x]`.
- Every public call of a rule (`call_*`, `@call_*`, `message_passing_*`) returns a
  `RuleResult`; `getresult` is the value, `getlogscale` its log scale. The engine uses
  `execute_rule`/`execute_rule_with_logscale` and never builds one; inlined,
  `getresult(message_passing_rule(...))` still allocates nothing (the routing gates).
- The v6 code is in the 6.x releases and in git.
- The rich displays, in MessagePassingRulesBase's repository (`svg.jl`, `display.jl`,
  `result_show.jl`), draw a node once, through
  `svg_node`, for two uses: the cards (`text/html`), themed by their CSS, and `drawing(x)`, a
  standalone SVG whose colours are presentation attributes from `SVG_LIGHT`. The card's CSS
  overrides those attributes, so a change of colour goes in `SVG_LIGHT` and the dark CSS palette.
- **Reactant** (`ext/ReactiveMPReactantExt.jl`): since Reactant 0.2.292 (EnzymeAD/Reactant.jl#3378),
  its overlays of `all` and `any` walk every field of their arguments, which throws on Rocket's
  linked lists (undefined placeholder fields, cycles). The extension marks the functions that build a
  graph or tell a variable's kind with `Reactant.@skip_rewrite_func`, in `__init__` (precompilation
  does not keep the marks); never a function on the path of a rule. `factornode`'s positional call is
  inlined into its caller, so `create_factornode` is marked too. Its test, `make test-reactant`, has its
  own environment and the workflow *Reactant check*, apart from `ci.yml`, so Reactant breaking does not
  block the suite; it compiles before running uncompiled, since `factornode`'s creation-plan
  cache would otherwise spare the trace the code under test. RxInfer marks its own functions in its
  `ReactantExt`.
- `src/fixes.jl` holds deliberate hot-fixes for upstream packages; it is expected to be
  empty when everything upstream has released.
