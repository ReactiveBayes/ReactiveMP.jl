# Phases — rule/node system rewrite

State tracker for the work described in `PLAN.md` (decisions) and `DISCUSSION.md`
(rationale). This file is the **volatile** one: it says where we are, not what we decided.

**Rule: update this file in the same commit as the change it describes.** Status claimed
without a diff alongside it is how a tracking file starts lying.

Branch: `refactor/rule-node-system-rewrite`

---

## Next action

**Phase 0 — the spike.** Nothing else should start until the devirtualization gate is
answered, because every later decision assumes it passes.

---

## Status at a glance

| # | Phase | Status |
|---|---|---|
| — | Design discussion, `PLAN.md`, `DISCUSSION.md` | **done** |
| 0 | Spike: devirtualization gate + syntax samples | **not started** |
| 1 | Circulate for external feedback | not started |
| 2 | Tooling migration on ReactiveMP | not started *(parallel with 1)* |
| 3 | `MessagePassingRulesBase` | not started |
| 4 | `MessagePassingRulesTestUtils` | not started |
| 5 | `StandardMessagePassingRules` | not started |
| 6 | `MessagePassingRulesApproximations` + node packages | not started |
| 7 | ReactiveMP engine rewrite | not started *(under-planned — wants its own design session)* |

---

## Phase 0 — Spike (throwaway)

**Goal:** answer the one question that cannot be walked back, before building anything.

Hand-written, no macros: target types, algorithm, context, a keyed-input
`message_passing_rule`, and three real rules — a simple BP rule, a structured VMP rule with
mixed `m[]`/`q[]`, and one with a variadic group.

**Exit criteria**
- [ ] `@code_typed`/JET show rule dispatch is static, with no dynamic dispatch
- [ ] a three-deep fallback chain compiles to the same code as a direct call
- [ ] ten representative rules written by hand in each candidate syntax, read side by side
- [ ] outbound-edge spelling decided
- [ ] where `algorithm` sits in the header decided
- [ ] **test the dependency language against the delta-node layouts** — express all three
      (default, CVI-projection) as declarations and see what does not fit. They are the
      hardest case, and the answer decides whether `AbstractDeltaNodeDependenciesLayout`
      (~684 lines) collapses and whether `CVIProjection` can ship as an extension
      (`PLAN.md` § CVI projection)

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
- [ ] `make test` = fast subset, `make test-all` = everything
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

Definition verification lands **before** Phase 5, not after: it is the difference between
checking ported rules against v6's output and checking them against the mathematics. Expect
it to surface rules that were already wrong.

---

## Phase 5 — `StandardMessagePassingRules`

**Goal:** standard distribution nodes plus arithmetic (`+`, `-`, `*`, dot).

**Exit criteria**
- [ ] JuliaSyntax-based migration tool, run with ReactiveMP v6 loaded as an oracle for
      `interfaces(fform)` (do **not** regex-guess on `_`)
- [ ] migrated per rule directory, diffs reviewed per directory
- [ ] canary passing: `NormalMixture((:m, k))` — indexed target + group + `where {N}` +
      aligned dependency
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
- [ ] delta node's built-in method set is now `{Unscented, Linearization}`; the
      "did you load `ExponentialFamilyProjection`?" diagnostic must be good
- [ ] `CVIProjection` ships as a weakdep extension of the Delta node package (assumes the
      Phase 0 layout result; if layouts do not collapse, it needs its own package instead)
- [ ] `MessagePassingRulesApproximations`: `Unscented`, `Linearization`, `CVI`, CVI
      projection, optimizers, `smoothRTS`, `approximations.jl`, `shared.jl`.
      **Standalone — must not depend on `MessagePassingRulesBase`.** Utilities that
      algorithms use, not algorithms. Deps: `ForwardDiff`, `DiffResults`, `Distributions`,
      `Random`, `LinearAlgebra` — no cubature package
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

**Closes open items:** #5.

---

## Open items

Tracked in `PLAN.md` § Open items. 8 listed, #8 resolved, so 7 live. Item #4 (ruleset axis) is the
weakest-supported — its piracy argument died with the empirical finding in
`DISCUSSION.md` §5.

## Structural note

Rules are pure functions of their inputs, so **Phases 3–5 need no engine at all**. A stall
on Phase 7 does not block anything else.
