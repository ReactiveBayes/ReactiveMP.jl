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
| — | Design discussion, `PLAN.md`, `DISCUSSION.md` | **done** |
| — | External design review; contradictions reconciled | **done** |
| P | Prep: baselines + disposition inventory | **not started** |
| 0 | Spike: dispatch gate, syntax samples, delta dependency semantics | **not started** |
| 1 | Circulate for external feedback | not started |
| 2 | Tooling migration on ReactiveMP | not started *(parallel with 1)* |
| 3 | `MessagePassingRulesBase` | not started |
| 4 | `MessagePassingRulesTestUtils` | not started |
| 5 | `StandardMessagePassingRules` | not started |
| 6 | `MessagePassingRulesApproximations` + node packages | not started |
| 4.5 | **Engine integration slice** — small end-to-end proof | not started |
| 7 | ReactiveMP engine rewrite | not started *(under-planned — wants its own design session)* |
| 8 | Release and downstream coordination | not started |

---

## Phase P — Prep

**Goal:** be able to measure regressions, and know what has to move.

**Exit criteria**
- [ ] contradictions across the three documents reconciled *(done)*
- [ ] **performance baseline captured** on v6 via the existing `benchmark/` + PkgBenchmark
      suites (`make bench`): compile latency, allocations, one full inference workload
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

**Goal:** answer the one question that cannot be walked back, before building anything.

Hand-written, no macros: target types, algorithm, context, a keyed-input
`message_passing_rule`, and three real rules — a simple BP rule, a structured VMP rule with
mixed `m[]`/`q[]`, and one with a variadic group.

**Exit criteria**
- [ ] `@code_typed`/JET show rule dispatch is static, with no dynamic dispatch
- [ ] **decide whether the ruleset axis exists at all** (open item #4) *before* writing a
      gate for it — its main justification died when the piracy check turned out to be
      vacuous for rules. If it stays: specify precedence, termination, and which failures
      permit falling through. **An exception raised inside a selected rule must propagate,
      never be treated as "try the next ruleset"**
- [ ] if the axis stays, a fallback chain adds no measurable routing overhead versus a
      direct call (equivalent dispatch behaviour + measured overhead, not byte-identical
      generated code — that is too brittle an acceptance criterion)
- [ ] ten representative rules written by hand in each candidate syntax, read side by side
- [ ] outbound-edge spelling decided
- [ ] where `algorithm` sits in the header decided
- [ ] **test the dependency language against the delta-node layouts** — express all
      **four** (default, known-inverse, CVI, CVI-projection) as declarations and see what
      does not fit. They are the
      hardest case, and the answer decides whether `AbstractDeltaNodeDependenciesLayout`
      collapses and whether `CVIProjection` can ship as an extension
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
- [ ] one worked `@allocate` example end to end
- [ ] the two hard context services as standalone calls (open item #12): the mixture switch
      rule with a product-and-log-scale service, and a delta rule using a captured function
      with fixed arguments — both with no graph construction and no Rocket
- [ ] measure, do not just assert: cold first invocation, warm execution, allocations, and
      specialization growth across variadic group sizes and heterogeneous input types
      (`NamedTuple` keys are invariant type parameters, so method-table pressure across many
      key sets is the specific risk)

**Closes open items:** #1 (syntax final form).

**If the gate fails:** the keyed-input design is wrong and `PLAN.md` needs reworking before
anything is built. That is the point of doing this first — cost is days, not months.

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
- [ ] built test-first throughout

**Closes open items:** #2, #3 (selector generality and factorisation keying — decide when
the language is real).

---

## Phase 4 — `MessagePassingRulesTestUtils`

**Goal:** test tooling as its own package, consumed via `[extras]`.

**Exit criteria**
- [ ] `@test_rules` successor: numerical output, type promotion (default on),
      `rule`/`rule!` agreement, optional non-allocating flag
- [ ] **node-definition verification** — reference update computed from `nodefunction`
      rather than from golden values
- [ ] registry-backed coverage check: every `RuleSpec`/`NodeSpec` has a test
- [ ] **migration checker**: runs a v6 and a v7 rule on identical inputs and asserts they
      agree — the tool that makes downstream (and agent-driven) migration verifiable

Definition verification lands **before** Phase 5, not after: it is the difference between
checking ported rules against v6's output and checking them against the mathematics. Expect
it to surface rules that were already wrong.

---

## Phase 4.5 — Engine integration slice

**Goal:** prove the rule/engine *interface* before porting hundreds of rules against it.

Phases 3–5 can be built without an engine because rules are pure functions. That does not
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
- [ ] delta node's built-in method set is now `{Unscented, Linearization}` — **the only
      capability regression in the plan**; accepted, but it needs (a) an explicit breaking
      entry in the release notes, not folded in with the renames, and (b) an error that
      names both the package to install and the method to switch to. First real customer
      for the registry-backed diagnostics
- [ ] `CVIProjection` ships as a weakdep extension of the Delta node package (assumes the
      Phase 0 layout result; if layouts do not collapse, it needs its own package instead)
- [ ] `MessagePassingRulesApproximations`: `Unscented`, `Linearization`, `smoothRTS`,
      `approximations.jl`, `shared.jl`. **Standalone — must not depend on
      `MessagePassingRulesBase`.** Utilities that algorithms use, not algorithms. Deps:
      `ForwardDiff`, `Distributions`, `Random`, `LinearAlgebra` — no cubature package, no
      `DiffResults` (it leaves with `cvi.jl`), no `Optim`
- [ ] API carried over as-is and prettified, **not redesigned**; `ctx` threaded where
      `cholinv` is currently global
- [ ] `ghcubature` moves to the Pólya node package along with `FastGaussQuadrature`
- [ ] confirm `Optim` no longer appears anywhere
- [ ] non-standard nodes spun out: Delta, Flow, Autoregressive, GP, BIFM, Pólya, …
- [ ] Pólya package carries the GPL-3 `PolyaGammaHybridSamplers`; ReactiveMP's MIT licence
      becomes honest again (see `PLAN.md` § Licensing)
- [ ] impure algorithms (BIFM, CVI) carry the `pure = false` marker

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
- [ ] pin the unexplained `reverse(...)` in mixture marginal wiring with a regression test
      *before* touching it

- [ ] explicit checks on scheduling order, annotations, retained values and free energy —
      not just numerical rule equality

**Closes open items:** none — the engine has no open item of its own; see the `PLAN.md`
list for live items.

---

## Phase 8 — Release and downstream coordination

A clean break removes compatibility shims; it does not remove release coordination.

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

Tracked in `PLAN.md` § Open items. 14 listed, #8 resolved, so 13 live.
Items #9–#14 came from external review and are API decisions blocking Phase 3. Item #4 (ruleset axis) is the
weakest-supported — its piracy argument died with the empirical finding in
`DISCUSSION.md` §5.

## Structural note

Rules are pure functions of their inputs, so **Phases 3–5 need no engine at all**. A stall
on Phase 7 does not block anything else.
