# Phases — rule/node system rewrite

State tracker for the work described in `PLAN.md` (decisions) and `DISCUSSION.md`
(rationale). This file is the **volatile** one: it says where we are, not what we decided.

**Rule: update this file in the same commit as the change it describes.** Status claimed
without a diff alongside it is how a tracking file starts lying.

Branch: `refactor/rule-node-system-rewrite`

---

## Next action

**Phase P**, then **Phase 0**. Phase P is cheap and makes everything after it measurable;
Phase 0 answers the questions that cannot be walked back.

---

## Status at a glance

| # | Phase | Status |
|---|---|---|
| — | Initial design documented in `PLAN.md`, `DISCUSSION.md` | **done; open decisions below** |
| — | External design review; contradictions reconciled | **done** |
| P | Prep: baselines + disposition inventory | **not started** |
| 0 | Spike: dispatch gate, syntax samples, delta dependency semantics | **not started** |
| 1 | Circulate for external feedback | not started |
| 2 | Tooling migration on ReactiveMP | not started *(parallel with 1)* |
| 3 | `MessagePassingRulesBase` | not started |
| 4 | `MessagePassingRulesTestUtils` | not started |
| 4.5 | **Engine integration slice** — small end-to-end proof | not started |
| 5 | `StandardMessagePassingRules` | not started |
| 6 | `MessagePassingRulesApproximations` + node packages | not started |
| 7 | ReactiveMP engine rewrite | not started *(under-planned — wants its own design session)* |
| 8 | Release and downstream coordination | not started |

---

## Phase P — Prep

**Goal:** know what has to move, and clear the rot that would otherwise be carried along.

**Exit criteria**
- [x] contradictions across the three documents reconciled; technical decisions and gates
      remain open as listed below
- [x] **performance baselining dropped from this phase.** The `benchmark/` suite,
      `scripts/bench.jl` and `make bench` are deleted rather than repaired: the suite only
      ever covered `DiscreteTransition`, its `ContinuousTransition` half was never included
      by `benchmark/rules/rules.jl` (and called `StableRNGs(42)`, a module rather than a
      constructor), its comparison path called a `PkgBenchmark` method through
      `BenchmarkTools`, no CI workflow ran it, and its output paths were gitignored so no
      result was ever retained. Performance verification belongs to
      **RxInferBenchmarks.jl** at implementation time, where there is a new engine to
      measure against
- [ ] **current Aqua ambiguity count measured**, so its cleanup can be budgeted separately
      from the new dispatch design
- [ ] **disposition inventory** (open item #14): every node, rule, extension, exported
      helper and engine hook assigned a destination or a deliberate deletion — including
      aliases, form constraints, fallbacks, callbacks, stream postprocessors, scoring
      helpers. Exported deletions get migration entries even when the answer is "no
      replacement"
- [ ] environment strategy for the v6/v7 comparison harness and before/after doctests —
      separate pinned environments if old and new constraints cannot coexist

---

## Phase 0 — Spike (throwaway)

**Goal:** test dispatch and dependency assumptions before committing to the API.

Hand-written, no macros: target types, algorithm, context, a keyed-input
`message_passing_rule`, and three real rules — a simple BP rule, a structured VMP rule with
mixed `m[]`/`q[]`, and one with a variadic group.

**Exit criteria**
- [ ] `@code_typed`/JET show the new routing machinery has no dynamic dispatch; assess rule
      bodies and user-supplied services separately
- [ ] record whether to include the ruleset axis (open item #4) or defer it pending a
      concrete use case. If included, specify precedence, termination and fallthrough
- [ ] **specify existing rule-fallback behavior regardless of the ruleset decision**:
      distinguish a missing rule from an exception inside a selected rule. Such an
      exception must propagate, never trigger fallback
- [ ] if the axis stays, a fallback chain adds no measurable routing overhead versus a
      direct call within the stated benchmark tolerance (equivalent dispatch behaviour +
      measured overhead, not byte-identical generated code)
- [ ] ten representative rules written by hand in each candidate syntax, read side by side
- [ ] outbound-edge spelling decided
- [ ] where `algorithm` sits in the header decided
- [ ] **test the dependency language against the delta-node layouts** — express all
      **four** (default, known-inverse, CVI, CVI-projection) as declarations and see what
      does not fit. Old CVI is a migration reference, not a surviving implementation
      requirement. These are the hardest cases, and the answer decides whether
      `AbstractDeltaNodeDependenciesLayout` collapses and whether `CVIProjection` can ship as an extension
      (`PLAN.md` § CVI projection)
- [ ] **test execution semantics, not just whether the dependency list can be expressed.**
      A declaration can name mathematically correct inputs and still produce a graph that
      stalls or updates in a different order. Layouts carry behaviour beyond input choice:
      `q_out` *aliases* the connected variable's marginal; static arguments gate execution
      while their values enter through the function proxy; initial values affect stream
      refresh; self-dependent updates need initialization. `dependencies.jl:35` records that
      changing refresh handling **changes free-energy trajectories and breaks strict
      FE-monotonicity**. So: execute minimal default, known-inverse and CVI-projection
      cases — including a delayed static input and an initialized feedback loop — and check
      emissions *and* numbers. Note `Mixture`'s `RequireMarginal` path is dead code that
      would `MethodError`, so one documented dependency mode has never actually run
- [ ] one worked allocation example end to end, using the intended `@allocate` lowering
      written by hand (the spike does not implement macros)
- [ ] the two hard context services as standalone calls (open item #12): the mixture switch
      rule with a product-and-log-scale service, and a delta rule using a captured function
      with fixed arguments — both with no graph construction and no Rocket
- [ ] measure, do not just assert: cold first invocation, warm execution, allocations, and
      specialization growth across variadic group sizes and heterogeneous input types
      (many key sets and input types may increase compiled specializations)

**Decision checkpoints:** #1 (syntax final form), #4 (include or defer rulesets), and
evidence for #12 (context services; the final contract is due in Phase 3).

**If a gate fails:** revise the affected dispatch, dependency or service design before
freezing the API. That is the point of doing this first — cost is days, not months.

---

## Phase 1 — Circulate

**Goal:** external feedback *before* the macro exists, because the macro is where effort
starts compounding and changing the surface afterwards means touching everything again.

**Exit criteria**
- [ ] `PLAN.md` + `DISCUSSION.md` + spike results shared
- [ ] go/no-go gate result posted as a GitHub issue for comment
- [ ] feedback triaged into `PLAN.md` edits or new open items

Use GitHub issues here, not files — humans comment on issues, agents read files.

---

## Phase 2 — Tooling migration *(can run in parallel with Phase 1)*

**Goal:** make every later session faster. Independent of the redesign, low risk.

**Exit criteria**
- [ ] `runtests.jl` filters by **name and tags**, not just filename (TestItemRunner already
      passes `(filename, name, tags)` to the filter — no package swap needed)
- [ ] tag taxonomy applied: `:rules`, `:nodes`, `:engine`, `:alloc`, `:slow`, `:quality`
- [ ] `make test` = fast subset, `make test-all` = everything — **the fast local default
      must not weaken full CI coverage**; CI runs everything
- [ ] Runic replaces JuliaFormatter
- [ ] Aqua `ambiguities` enabled
- [ ] Aqua `piracies` enabled — 3 known methods fixed or in `treat_as_own`
      (`uniform.jl:6,9`, `fixes.jl:12`; see `DISCUSSION.md` §5)
- [ ] `deps_compat`'s `check_extras` re-enabled
- [ ] `CLAUDE.md` "Running things" updated to match

---

## Phase 3 — `MessagePassingRulesBase`

**Goal:** the base package. Macros, types, dispatch, registry, dependency language. No rules.

**Exit criteria**
- [ ] `@define_factor_node` with variadic interface groups
- [ ] `@define_message_update_rule` / `@define_marginal_update_rule` / `@define_average_energy`
- [ ] `RuleSpec`/`NodeSpec` registry, per-module const + discovery (never `push!` into a
      shared global — precompilation hazard, see `PLAN.md`)
- [ ] dependency language with the four selectors + static-arity enforcement
- [ ] `RuleContext`, `buffer_like`, `@allocate`
- [ ] registry-backed errors; `check_rules()`, `check_rule_ambiguities()`
- [ ] CI assertion: `ExponentialFamily` absent from the dependency closure
- [ ] **resolve open items #9 (beliefs consumed vs entropy partition), #10 (buffer
      ownership), #11 (capability declaration for missing-extension diagnostics), #12
      (context service contracts), #13 (approximations protocol)** — all are API decisions
      that must land before the macro surface freezes
- [ ] **registry lifecycle test matrix**: fresh-process load after precompilation, both
      extension load orders, definitions in nested modules, supported interactive
      redefinition. Test duplicate signatures separately from ambiguous ones
- [ ] purity and RNG ownership contracts specified, including permitted output/scratch
      writes and the distinction between the audit policy and differentiation support
- [ ] built test-first throughout

**Decision checkpoints:** #2 (keep generalization deferred unless needed), #3 (factorisation
keying), and #9–#13 (resolve before the API freezes).

---

## Phase 4 — `MessagePassingRulesTestUtils`

**Goal:** test tooling as its own package, consumed via `[extras]`.

**Exit criteria**
- [ ] `@test_rules` successor: numerical output, type promotion (default on),
      `rule`/`rule!` agreement, optional non-allocating flag
- [ ] **node-definition verification** — reference update computed from `nodefunction`
      for a bounded initial subset, with separate shape and scale assertions
- [ ] registry-backed coverage check: every `RuleSpec`/`NodeSpec` has a test; record the
      actual selected rule so a fallback cannot conceal an untested specialization
- [ ] **migration checker**: runs a v6 and a v7 rule on identical inputs and asserts they
      agree — the tool that makes downstream (and agent-driven) migration verifiable
- [ ] disagreements with v6 investigated and recorded as migration bugs or deliberate
      mathematical corrections; analytic/finite-difference derivative checks cover both
      allocating and in-place paths

Definition verification lands **before** Phase 5, not after: it is the difference between
checking ported rules against v6's output and checking them against the mathematics. Expect
it to surface rules that were already wrong.

---

## Phase 4.5 — Engine integration slice

**Goal:** prove the rule/engine *interface* before porting hundreds of rules against it.

Rule kernels and test utilities can be developed without an engine. That does not
establish that their interface with the engine is correct — **this is the single largest
planning risk**, and the cheapest insurance is a small end-to-end proof first.

**Exit criteria**
- [ ] a working end-to-end inference over a handful of hand-ported rules covering: ordinary
      belief propagation, structured VMP, a mixture (variadic group), and a delta node
- [ ] free energy computed and compared against v6 on the same model
- [ ] annotations and log scales preserved (see the annotation gate in `PLAN.md`)
- [ ] a retained-value test: hold a message across several updates and confirm it is not
      mutated underneath you

Do not start Phase 5 until this passes.

---

## Phase 5 — `StandardMessagePassingRules`

**Goal:** standard distribution nodes plus arithmetic (`+`, `-`, `*`, dot).

**Exit criteria**
- [ ] JuliaSyntax-based migration tool, run with ReactiveMP v6 loaded as an oracle for
      `interfaces(fform)` (do **not** regex-guess on `_`)
- [ ] migrated per rule directory, diffs reviewed per directory
- [ ] canary passing: `NormalMixture((:m, k))` — indexed target + group + `where {N}` +
      aligned dependency
- [ ] **`MIGRATION.md` written *during* this phase, not after** — the mechanical rules are
      discovered while porting, and reconstructing them later leaves gaps exactly where the
      work was fiddly. Derived from the same source as the transform tool, with a test
      asserting the two agree
- [ ] every before/after pair in the guide is an executable doctest run by CI
- [ ] guide covers the untranslatable cases explicitly (raw `messages[i]` indexing, rules
      constructing graph objects, `meta`-as-mutable-workspace) and tells the reader — human
      or agent — to stop and ask rather than guess
- [ ] guide opens with a short preamble addressed to an agent: what to read, what never to
      guess, how to verify, when to stop
- [ ] hand-written cases done: `mixture/switch.jl`, the ~15 rules touching raw
      `messages[i]`/`marginals[i]`

---

## Phase 6 — Approximations and node packages

**Exit criteria**
- [ ] **delete first, package second** — `sphericalradial.jl`, `gausslaguerre.jl`,
      `importance.jl`, `laplace.jl` have no consumer in `src/`; remove them and their tests
      (skim the tests first, they may be the only record of intended behaviour)
- [ ] remove the superseded `cvi.jl` (`ProdCVI`/`CVI`), `delta/layouts/cvi.jl`,
      `rules/delta/cvi/*`, and with them `ReactiveMPOptimisersExt`, the `Optimisers`
      weakdep and `DiffResults`
- [ ] delta node's built-in method set is now `{Unscented, Linearization}` — **an accepted
      capability regression**, alongside exported deletions; it needs (a) an explicit breaking
      entry in the release notes, not folded in with the renames, and (b) an error that
      names both the package to install and the method to switch to. First real customer
      for the diagnostics and host-side capability metadata (open item #11)
- [ ] `CVIProjection` ships as a weakdep extension of the Delta node package (assumes the
      Phase 0 layout result; if layouts do not collapse, it needs its own package instead)
- [ ] `MessagePassingRulesApproximations`: `Unscented`, `Linearization`, `smoothRTS`,
      `approximations.jl`, `shared.jl`. **Standalone — must not depend on
      `MessagePassingRulesBase`.** Utilities that algorithms use, not algorithms. Deps:
      `ForwardDiff`, `Distributions`, `Random`, `LinearAlgebra` — no cubature package, no
      `DiffResults` (it leaves with `cvi.jl`), no `Optim`
- [ ] numerical API carried over without broader redesign; replace global `cholinv` calls
      through the minimal numerical protocol settled in Phase 3 (open item #13)
- [ ] `ghcubature` moves to the Pólya node package along with `FastGaussQuadrature`
- [ ] confirm `Optim` no longer appears anywhere
- [ ] non-standard nodes spun out: Delta, Flow, Autoregressive, GP, BIFM, Pólya, …
- [ ] Pólya package carries the GPL-3 `PolyaGammaHybridSamplers`; ReactiveMP's MIT licence
      becomes honest again (see `PLAN.md` § Licensing)
- [ ] surviving impure algorithms (BIFM and stateful projection algorithms) carry the
      `pure = false` marker under the agreed purity/RNG contract

## Phase 7 — ReactiveMP engine rewrite

**Under-planned. Wants its own design session before it starts.** `PLAN.md` treats this as
one line item; it is not.

Known scope, incomplete:
- [ ] mixture `activate!` demolition via variadic groups (~70–85% deletable)
- [ ] dependency-to-stream wiring for groups (`__collect_latest_updates` must collapse
      consecutive same-name interfaces)
- [ ] `Message`/`DeferredMessage` envelope changes (immutable `Message`, annotations as a
      type parameter)
- [ ] `EqualityChain` `BitVector` → `Vector{Bool}`
- [ ] engine diagnostics: `check_everything_pure`, `check_everything_inplace`, checked buffers
- [ ] preserve edge order when building clusters (GraphPPL factorisation indexes the
      original flat list — the single biggest correctness trap)
- [ ] plumb `EdgeLabel.index` through RxInfer instead of re-deriving group indices from
      neighbor position (open item #7)
- [ ] pin the unexplained `reverse(...)` in mixture marginal wiring with a regression test
      *before* touching it

- [ ] explicit checks on scheduling order, annotations, retained values and free energy —
      not just numerical rule equality

**Closes open items:** #6 (mixture regression pinned before rewriting) and #7 (edge indices
preserved through integration).

---

## Phase 8 — Release and downstream coordination

A clean break removes compatibility shims; it does not remove release coordination.
Start coordinated downstream CI as soon as compatible development revisions exist;
this phase requires it to pass for release, rather than being its first execution.

**Exit criteria**
- [ ] **strict coordinated downstream CI**: a job pinning mutually compatible revisions of
      the new packages and their consumers, in which `Pkg.Resolve.ResolverError` is a hard
      failure. Today `.github/workflows/IntegrationTest.yml` catches it and `exit(0)`s, so
      it would report green without running a single downstream test
- [ ] package registration order decided, compat bounds set, supported Julia versions agreed
- [ ] RxInfer's default package set updated
- [ ] documentation links across the three levels updated
- [ ] downstream migration readiness confirmed — `MIGRATION.md` exercised against a real
      external package (RxGP is the natural candidate)

---

## Open items

Tracked with stable numbers in `PLAN.md` § Open items. Of 14 items, #8 is resolved;
the remaining 13 are not all immediate blockers. #14 is Phase P inventory work. #1 is a
spike decision; #3 and #9–#13 must be settled before Phase 3's API freezes. #6 and #7 are
engine integration requirements. #2 remains deferred unless needed; #4 may be deferred
pending a concrete ruleset use case. #5 (Reactant/StableCholesky) belongs to a separate
effort and does not block this rewrite.

## Structural note

Rule kernels and test utilities can be developed independently of the engine. However,
**Phase 5 bulk migration is gated on Phase 4.5**, and release is gated on full engine and
downstream integration. Engine independence does not establish interface correctness.
