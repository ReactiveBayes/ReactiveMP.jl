# Phases — rule/node system rewrite

State tracker for the work described in `PLAN.md` (decisions) and `DISCUSSION.md`
(rationale). This file is the **volatile** one: it says where we are, not what we decided.

**Rule: update this file in the same commit as the change it describes.** Status claimed
without a diff alongside it is how a tracking file starts lying.

Branch: `refactor/rule-node-system-rewrite`

`file:line` citations in the three documents were re-verified against the code after the
Runic reformat (`3e3adae6`). A later reformat or edit moves them again, so re-check before
relying on one.

---

## Next action

**Phase 4.5 — the engine design session, then the first cut of the v7 engine.** Phase 4.5
was restructured on 2026-09-23 (`DISCUSSION.md` §3.18): there is no bridge into the v6
engine. The new engine is written directly in `src/`, built first for four slice cases, and
compared against fixtures recorded from v6. The design session comes before any code; its
agenda, the contracts it must honour and the v6 starting points are in § Phase 4.5 below.

Phases 0–4 are closed. `lib/MessagePassingRulesBase` is the rule system; `lib/MessagePassingRulesTestUtils`
is its test tooling; `compat/v6-comparison` holds the v6 oracle and the migration check.
The first finding the tooling produced is a real v6 bug — the variational
`NormalMeanVariance` rules use `E[v]` instead of `1/E[1/v]` for a non-point-mass `q_v`
(ReactiveMP.jl#669) — pinned as known, to be ported as a declared `:correction` in Phase 5.

**Picking this up on another machine.** Everything lives in the repository; nothing needed is
local. With Julia 1.10 and 1.13 installed (juliaup):

```bash
git switch refactor/rule-node-system-rewrite && git pull
make test-base                     # the base package (Julia on PATH)
make test-testutils                # TestUtils, developing the base at test time
julia +1.10 --startup-file=no --project=compat/v6-comparison -e 'using Pkg; Pkg.instantiate()'
julia +1.10 --startup-file=no --project=compat/v6-comparison compat/v6-comparison/check.jl
```

Then read, in order: `CLAUDE.md`, `PLAN.md`, `DISCUSSION.md` §4 *Corrections* and §3.14–3.17,
and this file's § Phase 4.5. Working conventions established so far: one commit per step,
failing test first, `PHASES.md` and `CHANGELOG.md` updated in the same commit, descriptive
names rather than generic ones, and no comments that only narrate.

---

## Status at a glance

| # | Phase | Status |
|---|---|---|
| — | Initial design documented in `PLAN.md`, `DISCUSSION.md` | **done; open decisions below** |
| — | External design review; contradictions reconciled | **done** |
| P | Prep: disposition inventory, layout and environment decisions | **done** |
| 0 | Spike: dispatch gate, syntax samples, delta dependency semantics | **done** |
| 1 | Circulate for external feedback | **done** *(internally)* |
| 2 | Tooling migration on ReactiveMP | **done** |
| 3 | `MessagePassingRulesBase` | **done** |
| 4 | `MessagePassingRulesTestUtils` | **done** |
| 4.5 | **Engine design and first cut** — the real v7 engine, for four slice cases *(absorbs the start of 7)* | not started; next is the design session |
| 5 | `StandardMessagePassingRules` | not started |
| 6 | `MessagePassingRulesApproximations` + node packages | not started |
| 7 | Complete the engine — remaining nodes, diagnostics, RxInfer plumbing | not started |
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
- [x] **current Aqua ambiguity count measured**: **322**, so its cleanup can be budgeted
      separately from the new dispatch design. Breakdown recorded under Phase 2
- [x] **disposition inventory** (open item #14): every node, rule, extension, exported
      helper and engine hook assigned a destination or a deliberate deletion — including
      aliases, form constraints, fallbacks, callbacks, stream postprocessors, scoring
      helpers. Exported deletions get migration entries even when the answer is "no
      replacement"
  - [x] tooling: `scripts/inventory.jl --generate | --check` and `INVENTORY.md`.
        `--generate` enumerates from the live package and **preserves decisions already
        made**, so it is re-runnable; `--check` fails on a missing entity, an `undecided`
        destination, an invalid destination, a stale row, or an exported deletion with no
        migration note
  - [x] enumeration complete and located: **231 entities** — 49 nodes, 165 exported
        symbols, 8 engine-hook families, 2 extensions, 7 rule-level exceptions. Rules
        inherit their node's destination, so only the rules that cannot are listed
  - [x] **231 destinations decided**; `--check` passes. Totals: `standard` 58,
        `base` 43, `engine` 32, `delete` 21, `node:Flow` 16, `node:Delta` 11,
        `models` 11, `node:Polya` 10, `node:Autoregressive` 9, `approximations` 7,
        `node:ContinuousTransition` 5, `node:BIFM` 5, `node:DiscreteTransition` 3.
        `StandardMessagePassingRules` is distributions, arithmetic, logic and mixtures;
        domain-specific models (GCV, Probit, SoftDot, GaussianCoupling) go to a separate
        package **that is not yet named**, so the token is `models`
  - [x] `--check` wired into CI as `test/inventory_tests.jl`, tagged `:quality`.
        **Caveat:** no CI job *names* it — `julia-actions/julia-runtest` calls `Pkg.test()`
        with empty `ARGS`, so `runtests.jl` applies no filter and the item rides along inside
        the general test job. That is sufficient today and becomes load-bearing at Phase 2:
        a `make test` fast subset that drops `:quality` would silently un-gate the inventory.
        **Closed in Phase 2:** the fast subset drops only `:slow`, and CI sets `TEST_ALL=true`,
        so the inventory gate runs everywhere
- [x] **repository layout decided: monorepo under `lib/`, split at Phase 6.** Boundaries
      were still moving (#9–#13 were then unresolved API decisions), and a cross-package change is
      one commit in a monorepo versus two pull requests and a dev-pin across repositories.
      Pay the split cost once, at a known gate. See `PLAN.md` § Repository layout
- [x] **Julia floor decided: stays at 1.10.** Nothing in the design requires more. The
      `ScopedValue` in an earlier draft of § Dispatch axes was never necessary — the
      context is an ordinary object passed into the rules, most likely held by
      `MessageMapping`, and a plain default argument gives the same behaviour
- [x] environment strategy for the v6/v7 comparison harness and before/after doctests
  - [x] **one-process rule comparison strategy selected.** The new rule packages are
        differently named and do not depend on ReactiveMP. The stubs can coexist with
        `ReactiveMP@6.5.0`; Phase 3 must re-check resolution when real inter-package and
        numerical dependencies are added. The committed comparison environment currently
        pins v6 only. **Standing constraint:** the joint environment requires shared
        BayesBase and ExponentialFamily additions to stay
        within their current caret bounds (`BayesBase = "1.5"`, `ExponentialFamily = "2.5.0"`),
        so BayesBase work for this rewrite must ship as non-breaking 1.x releases
  - [x] **separate engine comparison strategy:** ReactiveMP v7 and v6 have the same UUID
        and cannot be loaded as two versions in one process. Use separate pinned processes
        or saved fixtures from full v6 inference runs. Phase 4's rule outputs alone cannot
        establish engine scheduling, free-energy trajectories or retained-value behavior
  - [x] `compat/v6-comparison/` created and verified: it instantiates and resolves
        ReactiveMP **from the registry**, not from this checkout (checked via `pathof`),
        which is what keeps it valid once the local copy becomes v7. Its `Manifest.toml`
        is un-ignored so the pin is reproducible
        **Review correction:** the original manifest was generated on Julia 1.13 and failed
        to load on 1.10 (`PrecompileTools`: `StaticData` undefined). It has been regenerated
        on Julia 1.10.12 and v6 loads successfully there; see `compat/v6-comparison/README.md`
  - [x] `lib/` skeleton created — four `Project.toml` stubs plus empty modules, all
        instantiating. Cross-package `[deps]` are deliberately left out for now: the
        packages are unregistered and `[sources]` needs Julia 1.11 while our floor is 1.10,
        so Phase 3 onwards wires them with `Pkg.develop(path = ...)` and commits the
        Manifest. Documented in `lib/README.md`. *(Superseded in Phase 4: the sibling is
        listed in `[deps]` and developed at test time, and no Manifest under `lib/` is
        committed — see `lib/README.md`.)*

---

## Phase 0 — Spike (throwaway) — **DONE**

**Goal:** test dispatch and dependency assumptions before committing to the API.

**The spike has been deleted, as planned.** The `spike/...` paths cited below are historical:
the whole tree is present at `81822c57`, so `git show 81822c57:spike/README.md` is the
way in. Findings, measurements and verdicts are in `DISCUSSION.md` §3.15 — that is the
document to read, not this one, which only records that the criteria were met.

Hand-written, no macros: target types, algorithm, context, a keyed-input
`message_passing_rule`, and three real rules — a simple BP rule, a structured VMP rule with
mixed `m[]`/`q[]`, and one with a variadic group.

**Exit criteria**
- [x] `@code_typed`/JET show the new routing machinery has no dynamic dispatch; assess rule
      bodies and user-supplied services separately. **Done** — `spike/dispatch/03_devirt.jl`
      measures routing on a body that allocates nothing, so routing and body costs are never
      mixed; JET reports nothing. Services are exercised separately in `06_services.jl`
- [x] record whether to include the ruleset axis (open item #4) or defer it pending a
      concrete use case. **DEFER** — `spike/dispatch/04_fallback.jl` shows a downstream
      package defining its own algorithm and getting its own rule for a *standard* node and
      edge, with no shadowing and no ambiguity, because the algorithm is part of the
      signature. The piracy argument for the axis was already dead (`DISCUSSION.md` §5).
      Nothing in tree needs scoped rule tables, and adding the axis later is a new keyword
      rather than a resurfacing
- [x] **specify existing rule-fallback behavior regardless of the ruleset decision**:
      distinguish a missing rule from an exception inside a selected rule. Such an
      exception must propagate, never trigger fallback.
      **Contract: resolution is a separate, total function.** `find_rule` returns a
      `RuleSpec` or a `RuleNotFound`; it never throws and never runs anything. The fallback
      is consulted on the `RuleNotFound` branch only, which is decided *before* any body
      runs, so there is no `try` anywhere near the body and an exception from a selected rule
      cannot reach the fallback even deliberately. A `try`/`catch` around execution would get
      this wrong silently, by turning a broken rule into a missing one. Applies uniformly to
      message rules, marginal rules and average energy — **v6's asymmetry is removed**, where
      `rule` returns a sentinel and `marginalrule` throws, so marginal rules cannot have a
      fallback at all for no stated reason. Verified in `spike/dispatch/04_fallback.jl`
- [x] if the axis stays, a fallback chain adds no measurable routing overhead versus a
      direct call. **Moot — the axis is deferred**, so there is no chain to measure. Recorded
      rather than silently dropped: if #4 is ever reopened, this gate comes with it, and the
      requirement is equivalent dispatch behaviour plus measured overhead, not
      byte-identical generated code. **No fixed tolerance is set, deliberately** — an earlier
      wording appealed to "the stated benchmark tolerance", which no document ever stated
- [x] **rule syntax decided** (open item #1, resolved ahead of the spike): fully
      keyword-based macro, body an ordinary lambda over a real `args` object, symbols
      throughout (`towards = :out`, `m[:μ]`, `interfaces = [:out, ...]`), group members
      `q[:p][k]`, indexed targets `(:m, k)`, body slots
      `(output, algo, ctx, args, ann, node)` in canonical order *(later reduced to five: the
      node moved into `ctx.node` at the Phase 3 sign-off, #12)*, dispatch carried by the
      `algorithm` keyword, `@allocate`/`@logscale` deleted. See `PLAN.md` § Rule surface and
      `DISCUSSION.md` §3.14
- [x] **`RuleSpec` representation decided** (resolved ahead of the spike): **no type
      parameters at all** — `body::Function`, `prealloc::Function`, `inplace::Bool`,
      `pure::Bool`, source/file/line and the registry metadata, all ordinary fields. The
      deciding argument is `find_rule`: a parameterised spec makes every rule a distinct type,
      so a lookup that cannot statically pin down which rule fires returns a *union* rather
      than one type, and that degrades silently. The plain struct is type-stable by
      construction. The indirect call this costs is accepted and revisited later with real
      rules; Phase 0 supplies the number. See `DISCUSSION.md` §3.14
- [x] **ten representative rules written by hand** and read side by side —
      `spike/dispatch/02_rules.jl`, each shown as the surface a user writes plus the form the
      macro would emit. Covers the trivial BP case, an annotating rule, the `meta::Any`
      arithmetic catch-all, the `NormalMixture((:m, k))` canary, a variadic group, the
      mixture-switch context service, delta with and without a known inverse, a marginal rule
      over a structural cluster, an in-place rule, and an average energy.
      **One finding.** The canonical body slots `(output, algo, ctx, args, ann, node)` have no
      target, but an indexed target `towards = (:m, k)` has to bind `k`, which is a runtime
      value the lowered body cannot close over. Resolution: thread `target` to every body and
      let the macro emit `k = index(target)` as an ordinary binding when the declaration names
      an index. It stays out of the user-facing slot list — writing `k` is how you ask for it
- [x] **the devirtualization gate must run through the `RuleSpec`**, not only through
      dispatch, and it must **report numbers rather than pass or fail**. `RuleSpec` carries no
      type parameters (`DISCUSSION.md` §3.14), so the body is reached through a `::Function`
      field and the indirect call is accepted by decision, to be revisited with real rules.
      What Phase 0 owes is the cost, measured under the real spec:
      - a call site that can reach exactly one rule — expected to fold away entirely;
      - a call site that can reach several — where the indirect call actually appears;
      - the same two figures for the parameterised alternative, so the trade has a number
        attached if it is ever reopened.
      **Measured** — `spike/dispatch/03_devirt.jl`, results under `spike/results/`:

      | | 1.10.12 | 1.13.0 |
      |---|---|---|
      | `args.m[:sym]`, `args.q[:p][k]` | 0, inferred | 0, inferred |
      | routing, call site reaching one rule | **0**, `Float64` | **0**, `Float64` |
      | routing, call site reaching two rules | 48, `Any` | 0, `Any` |
      | same, parameterised spec, two rules | 32, `Float64` | 0, `Float64` |
      | `NormalMeanVariance(:out)` end to end | **0** | **0** |
      | `typeof(+)(:in2)` end to end | **0** | **0** |

      So the indirect call the decision accepts costs **nothing on the ordinary path** — a
      factor node's form is fixed, so its call sites reach one rule — and 48 bytes on the
      floor only where resolution is genuinely ambiguous, which is where type information has
      already been lost upstream. On 1.13 even that is free. `find_rule` returns a concrete
      `RuleSpec` in every case. JET reports nothing on the routing.
      **Four ways to measure this wrongly, all of them hit while building the gate**: a
      non-`const` global (+16), a varargs helper that splats (+48), a spec constructed inline
      in an inlinable `find_rule` (constant-folded to 0 for *every* representation), and
      closing over the node in a loop so it is a `DataType` rather than `Type{Node}` (goes
      dynamic, reports `Any`). The gate caught two of them by failing
- [x] **test the dependency language against the delta-node layouts** — express all
      **four** (default, known-inverse, CVI, CVI-projection) as declarations and see what
      does not fit. Old CVI is a migration reference, not a surviving implementation
      requirement. These are the hardest cases, and the answer decides whether
      `AbstractDeltaNodeDependenciesLayout` collapses and whether `CVIProjection` can ship as an extension
      (`PLAN.md` § CVI projection).
      Located: `DeltaFnDefaultRuleLayout` (`delta/layouts/default.jl:18`),
      `DeltaFnDefaultKnownInverseRuleLayout` (`default.jl:234`),
      `CVIApproximationDeltaFnRuleLayout` (`delta/layouts/cvi.jl:16`) and
      `CVIProjectionApproximationDeltaFnRuleLayout`
      (`ext/ReactiveMPProjectionExt/layout/cvi_projection.jl:35`), each implementing or
      delegating `deltafn_apply_layout` over the four slots `q_out`, `q_ins`, `m_out`,
      `m_in`. Read across them, the layouts differ *only* in which messages and marginals
      each slot consumes — which is the hypothesis holding. Two things in the same files are
      **not** dependency choices, and are where it breaks if it breaks: `with_statics`
      (`default.jl:22-44`), which gates execution on const/data inputs whose values arrive
      out-of-band through the function proxy, and the `N === 1` compile-time branch
      (`default.jl:321-327`). `q_out` "mirrors the variable marginal" (`default.jl:47-64`) is
      stream aliasing — topology, not a rule input.
      **VERDICT: the hypothesis holds for input selection, and only for that.**
      `spike/semantics/08_layouts.jl` writes all four out as declarations and they differ in
      exactly one respect — which messages and marginals each slot consumes. Three things do
      not fit and are not dependency choices: static gating, the `N === 1` empty-group
      branch, and `q_out` aliasing. So the collapse is **real but partial**: dependencies
      absorb the input selection, and those three need explicit support in
      `MessagePassingRulesBase` or they land back in the engine. `CVIProjection` can ship as
      a Delta-package extension on that condition
- [x] **test execution semantics, not just whether the dependency list can be expressed.**
      A declaration can name mathematically correct inputs and still produce a graph that
      stalls or updates in a different order. Layouts carry behaviour beyond input choice:
      `q_out` *aliases* the connected variable's marginal; static arguments gate execution
      while their values enter through the function proxy; initial values affect stream
      refresh; self-dependent updates need initialization. `dependencies.jl:35` records that
      changing refresh handling **changes free-energy trajectories and breaks strict
      FE-monotonicity**. So: execute minimal default, known-inverse and CVI-projection
      cases — including a delayed static input and an initialized feedback loop — and check
      emissions *and* numbers. Note `Mixture`'s `RequireMarginal` path is dead code that
      would `MethodError`, so one documented dependency mode has never actually run —
      confirmed: `mixture.jl:146` takes three positional arguments while its only caller
      `with_functional_dependencies` (`dependencies.jl:119-126`) passes four, so dispatch
      falls through to the generic method, whose first statement is
      `getlocalclusters(factornode)` — and `MixtureNode` has no such method.
      **Done** — `spike/semantics/09_execution.jl` runs the live v6 engine on minimal
      default, known-inverse, static-input and Unscented cases and captures emission order
      *and* numbers as fixtures. The decisive result is the static one: **0 emissions before
      the static input arrives, 2 after**, which is `with_statics` gating execution rather
      than selecting inputs — a declaration that only names inputs cannot express it
- [x] one worked allocation example end to end, using the intended `preallocate` lowering
      written by hand (the spike does not implement macros). `spike/dispatch/05_allocate.jl`:
      `rule` and `rule!` agree numerically, the kernel with a provided buffer allocates
      **0 bytes** while the allocating form allocates 176 (its buffer — in-place is not the
      same property as non-allocating), the body uses `output` and `args.m[:out]` together,
      and a wrong `output` is a `MethodError` from Julia rather than from macro analysis
- [x] the two hard context services as standalone calls (open item #12): the mixture switch
      rule with a product-and-log-scale service, and a delta rule using a captured function
      with fixed arguments — both with no graph construction and no Rocket.
      `spike/dispatch/06_services.jl` turns both into signatures:
      `product : (left, right) -> (dist, logscale::Real)` and
      `nodefn : (ctx, target) -> callable of the free arguments`, neither carrying an engine
      type. **One finding: incoming annotations have no declared route.** The switch rule
      needs the log scales that *arrived* with its messages, but `args` holds message data and
      `ann` is an output sink. Recommend a parallel accessor keyed like `m` —
      `args.ann_in[:out]` *(not adopted: at the Phase 3 sign-off `ann` became two-way instead,
      #12)* — kept out of dispatch, since an annotation must not select the
      mathematics
- [x] measure, do not just assert: cold first invocation, warm execution, allocations, and
      specialization growth across variadic group sizes and heterogeneous input types.
      `spike/dispatch/07_measure.jl`, on the 1.10 floor: cold **5.8 ms** to compile one rule,
      warm **1.33 ns** at 0 allocations, and `Float64`/`Float32`/`BigFloat` all propagate
      without widening or allocation — the property that makes `ForwardDiff.Dual` work
      through a rule. Specialization growth: six group sizes add 20 specializations, and
      growth is **multiplicative in (group size × element type)**. That is a property of the
      tuple rather than of the new design — `ManyOf{N,T}` has it today — and it is the price
      of the statically known group arity `PLAN.md` § Dependencies already requires. It is
      the number to watch if compile time becomes the complaint in Phase 5

**Decision checkpoints:** #4 (include or defer rulesets) and
evidence for #12 (context services; the final contract is due in Phase 3).

**If a gate fails:** revise the affected dispatch, dependency or service design before
freezing the API. That is the point of doing this first — cost is days, not months.

---

## Phase 1 — Circulate — **DONE**

**Goal:** external feedback *before* the macro exists, because the macro is where effort
starts compounding and changing the surface afterwards means touching everything again.

**Closed internally.** The review happened directly rather than through a GitHub issue, so
there is no issue to link and none is needed. The corrections it produced are already in the
documents — `DISCUSSION.md` §4 *Corrections* is largely the record of it, and the
review-driven open items #9–#13 came out of it.

- [x] `PLAN.md` + `DISCUSSION.md` + spike results shared
- [x] go/no-go gate result reviewed *(internally, not as an issue)*
- [x] feedback triaged into `PLAN.md` edits or new open items

---

## Phase 2 — Tooling migration — **DONE**

**Goal:** make every later session faster. Independent of the redesign, low risk.

Seven of the eight items are done. The eighth, Aqua's `ambiguities`, is **deliberately left
off** with the reasoning and the measured baseline recorded below — it is a decision, not an
omission.

**Exit criteria**
- [x] `runtests.jl` filters by **name and tags**, not just filename (TestItemRunner already
      passes `(filename, name, tags)` to the filter — no package swap needed). Three kinds of
      `test_args` entry, composable: a path (`rules:beta:out`, unchanged), `tag:<name>` and
      `name:<text>`. Same kind OR'ed, different kinds AND'ed
- [x] tag taxonomy applied: `:rules` (196), `:nodes` (84), `:engine` (133 — everything that
      is not a rule or node test), `:alloc` (6, the items asserting allocation counts),
      `:quality` (1, the inventory gate). *(Counts corrected in the post-Phase-4 audit: the
      figures first recorded here, 194/83/130, summed to 408, not 414, and were already
      wrong at the tagging commit `80be2dcd`.)* **All 414 items carry a tag; none is `:slow` yet** —
      nothing has been measured as slow, so nothing claims to be. Tagging one later removes it
      from `make test` and leaves it in `make test-all` and CI
- [x] `make test` = fast subset, `make test-all` = everything — **the fast local default
      must not weaken full CI coverage**; CI runs everything. Enforced rather than intended:
      `ci.yml` sets `TEST_ALL=true`, so a `:slow` tag changes what a developer runs locally
      and never what CI runs
- [x] Runic replaces JuliaFormatter. Zero-config, so `.JuliaFormatter.toml` and its 27 style
      options are deleted — there is no style file left for CI and contributors to disagree
      through. **Measured: byte-identical output on Julia 1.10 and 1.13**, which JuliaFormatter
      could not manage; its output moved with the Julia minor version through JuliaSyntax,
      which is why `FormatCheck.yml` had to pin its Julia to the newest version in the test
      matrix. That pin is now the floor and matches `scripts/Manifest.toml`. Runic's own
      version stays pinned, since its output may change between releases. Reformatted 389 of
      518 files; `docs/` stays excluded, as before, and no docstring or doctest line was
      touched
- [ ] Aqua `ambiguities` — **deliberately left off; revisit after the split.** Re-measured on
      this branch (Julia 1.13, `Aqua.detect_ambiguities(ReactiveMP; recursive = true)`):
      **322 pairs**, identical to the Phase P baseline. Attributing each pair to the
      ReactiveMP files on either side:

      | pairs touching | file | character |
      |---|---|---|
      | 119 | `src/helpers/algebra/permutation_matrix.jl` | custom array types declaring `*`/`dot` against bare `AbstractMatrix`/`AbstractVector`, colliding with `ArrayLayouts`, `PDMats`, `FillArrays` and `LinearAlgebra` |
      | 85 | `src/helpers/algebra/standard_basis_vector.jl` | same shape |
      | 71 | `src/helpers/algebra/companion_matrix.jl` | same shape — and `CompanionMatrix` has **zero references anywhere in `src/` or `test/`**, so this is entirely dead weight |
      | 27 / 25 | `delta.jl` / `rule.jl` | **one** repeated shape: the delta catch-all against the `meta::Any` arithmetic rules. The only category the new dispatch design claims to eliminate. **27 distinct pairs, not 52** — 25 of them touch both files. And `rule.jl` here is the `@marginalrule` *template* (`rule.jl:406`), where every generated method reports its location, so it names the arithmetic rules in `src/rules/{addition,subtraction,multiplication}/`, not code in `rule.jl` (see `DISCUSSION.md` §5) |
      | 23 | `src/fixes.jl` | deliberate upstream hot-fixes; they leave when upstream releases |
      | 11 | `nodes/predefined/uninformative.jl` | `prod` for `Uninformative` against BayesBase's `PreserveTypeProd` |

      (A pair is counted against both files it touches, so the column does not sum to 322.)

      **Why not now.** Three files account for the large majority, `INVENTORY.md` already
      sends all three out of this package with Flow/AR and marks `CompanionMatrix` for
      deletion, and the 27 rule-dispatch pairs are what the rewrite removes by construction.
      Cleaning them here is work on code that is leaving, and a ratchet on a number that is
      about to move on its own would mostly measure the split rather than any regression.
      Revisit once Phases 5–6 have moved the rules and the algebra helpers out; the count to
      beat is recorded above.
- [x] Aqua `piracies` enabled — the 3 known methods declared via
      `treat_as_own = [Distributions.Uniform, ForwardDiff.Dual]`, after which **zero pirates
      remain** (measured). Both are documented where they are defined: the `Uniform`×`Beta`
      product (`uniform.jl:6,9`) is a mathematical special case that arguably belongs in
      ExponentialFamily, and the `dot` overload (`fixes.jl:12`) leaves when `src/fixes.jl`
      does
- [x] `deps_compat`'s `check_extras` re-enabled. It required two things: **compat bounds for
      every `[extras]` entry**, not only the runtime deps, and dropping three extras that
      nothing used — `Coverage`, `Dates` and `Distributed` (`Logging` looked unused too, but a
      multi-line import hid it; it stays)
- [x] `CLAUDE.md` "Running things" updated to match

---

## Phase 3 — `MessagePassingRulesBase`

**Goal:** the base package. Macros, types, dispatch, registry, dependency language. No rules.

### Entry brief — signed off 2026-09-22

The items gating the API freeze, each with the evidence gathered from the code and the
outcome of the sign-off. **The decisions are recorded in `PLAN.md` § Open items**, which is
the document of record; this section keeps the evidence and says what changed from the
proposal. Citations are as of `545425a2`.

| item | outcome |
|---|---|
| #3 group selection keying | **accepted** as proposed |
| #9 consumed vs. partition | **accepted** as proposed |
| #10 buffer ownership | **accepted, strengthened**: storage reuse is unspecified engine-internal behaviour; outsiders copy, outsider getters copy by default, `InputArgumentsAnnotations` deep-copies |
| #11 capability diagnostic | **proposal rejected**: keep the existing `is_delta_node_compatible` guard, no static table |
| #12 context services | **changed**: `ann` carries incoming annotations too (no `args.ann_in`); the node moves into `ctx.node` and the `node` slot is dropped; `nodefn` is not a service. The missing-input sub-point was left proposed at the sign-off and confirmed, exactly as v6, when Phase 3 closed |
| #13 approximations protocol | **parked** by the user; no proposal stands |
| RNG ownership | **accepted** as proposed |

**#3 — group selection keyed per target, or per (target, factorisation)?**
- *Evidence.* Selection is keyed per target today; the only factorisation input is
  `clusterindex` (`dependencies.jl`). No in-tree rule chooses group members from the
  factorisation: `NormalMixture`/`GammaMixture` reject anything but mean-field
  (`normal_mixture.jl:78-82`) and hard-code members by index, and `Mixture`'s `factornode`
  ignores the factorisation entirely (`mixture.jl:73-96`).
- *Proposed.* Keep it per target. Dependencies belong to the **algorithm**, so selection that
  genuinely varies with factorisation is a distinct algorithm, not a new syntax axis. That
  keeps the one-way door closed without building anything.

**#9 — beliefs consumed vs. the partition whose entropy is counted**
- *Evidence.* Free energy iterates only the cluster marginals
  (`get_node_local_marginals`, `score/node.jl:92`). `RequireMarginal` already adds a consumed
  marginal that is never a cluster (`FactorNodeLocalMarginal`, `dependencies.jl:300-320`), and
  never checks whether the edge is already inside a joint cluster. `ContinuousTransition`'s
  default dependencies (`continuous_transition.jl:97`) make `:a` consume `q_a` with no cluster.
  Joint names follow the cluster tuple, which GraphPPL sorts ascending, so they are always in
  interface-declaration order; `q_y_x` appears 16 times, `q_x_y` never, and no rule has a
  reversed twin. There is no permutation logic anywhere.
- *Proposed.* Two separate declarations: **consumed** (a dependency's right-hand side) and
  **partition** (derived from the factorisation, or declared by the algorithm). An auxiliary
  marginal is consumed and never scored. `q[:a, :b]` must list members in interface-declaration
  order, and `check_rules()` rejects any other order at definition time rather than permuting
  the joint. A user factorisation that conflicts with an algorithm's declared partition is an
  activation-time error naming the algorithm.

**#10 — buffer ownership**
- *Evidence.* Every retainer holds a reference, never a copy: `DeferredMessage.cache`
  (`message.jl:451,492`, never cleared), the `RecentSubject` behind every variable's streams,
  equality-chain caches (`equality.jl:37-38,144`; invalidation flips a bit and keeps the old
  message), `InputArgumentsAnnotations`, which stores inputs *and* result and grows through
  products (`input_arguments.jl:74-108`), callback events a user handler may keep,
  `PendingScheduler` queues, and `CVIProjection`'s mutable proposal container
  (`ext/ReactiveMPProjectionExt/rules/marginals.jl:222`). v6 is safe only because rules
  allocate fresh results.
- *Proposed.* Phase 3 defines `preallocate` and `rule!` and nothing about reuse. Every result
  the engine publishes is an **owned snapshot**; reusing a buffer is illegal until the engine
  opts an edge in, and the eligibility rule — which must exclude any edge feeding a retainer
  above — is specified in Phase 4.5/7. The base API freezes without committing the engine.

**#11 — capability metadata for the missing-extension diagnostic**
- *Evidence.* The only "load X" message today is a hand-written `error`
  (`cvi_projection.jl:138-140`). The only registered error hint is a `MethodError` hint for
  unsupported factor nodes and missing callback handlers (`ReactiveMP.jl:92`), not
  extensions. The positional `DeltaMeta{M, I}(…)` constructor bypasses the compatibility check.
- *Proposed.* The base package provides a static **capability declaration**,
  `(node, algorithm type) → (package to install, alternative methods)`, emitted by the host
  package (Delta), not by the extension. `find_rule` consults it on the `RuleNotFound` branch
  to produce the actionable error. `CVIProjection` stays a type defined in the host, so the
  declaration can name it before the extension loads.

**#12 — context service contracts and incoming annotations**
- *Evidence.* `rules/mixture/switch.jl` reaches the engine only for a product with log
  scale, and it *requires* every incoming message to carry `:logscale` — a `KeyError`
  otherwise (`logscale.jl:68-69`). The 12 `getnodefn` rules (plus 3 in `ext/`) only ever fetch
  `f` or its inverse; nothing calls `getnode()`. There are 46 FastCholesky calls in `src/`
  outside `approximations/`. On the missing-input path, pre-rule annotation processors run,
  the body and post-rule processors do not (`message.jl:678-728`).
- *Proposed* *(changed at the sign-off — see the table above: no `args.ann_in`, `ann` is
  two-way; `nodefn` is not a service; the services are `node`, `product`, `linalg`, `rng`,
  `context.jl:23`)*. Adopt `args.ann_in[:sym]`, keyed exactly like `m` and never dispatched on. Four
  services — `product`, `nodefn`, `linalg`, `rng` — declared with `ctx = (...)`. The concrete
  `ctx` type may be parameterised so services specialise; "non-dispatching" means it never
  selects the mathematics. The missing-input path skips both the body and the post-rule
  processors, as v6 does, and a test pins it.

**#13 — the approximations package's numerical protocol**
- *Evidence.* In the files that survive, only three linear-algebra sites remain:
  `unscented.jl:318` (`cholsqrt`) and `rts.jl:20,21` (`cholinv`). `smoothRTS` and
  `local_linearization` take no method argument, so nothing can be threaded through them today.
- *Proposed.* A duck-typed protocol **owned by the approximations package**:
  `approx_cholsqrt(strategy, A)` and `approx_cholinv(strategy, A)`, defaulting to FastCholesky.
  The strategy travels as a field on `Unscented`/`Linearization`; `smoothRTS` gains a trailing
  `strategy` argument. The base package's `linalg` service satisfies the protocol by defining
  methods, never through a dependency in either direction.

**RNG ownership (prerequisite for the purity contract)**
- *Evidence.* RNGs live in method and meta objects (`cvi_projection.jl:117`,
  `binomial_polya.jl:25`), are never reseeded or reset, and fall back to the global RNG
  elsewhere.
- *Proposed.* The RNG comes from `ctx.rng` and is owned by the caller. An algorithm that holds
  its own RNG is `pure = false`.

**Exit criteria**
- [x] `@define_factor_node` with variadic interface groups — several, non-trailing, and
      interface names containing underscores (`nodes:*` tests)
- [x] `@define_message_update_rule` / `@define_marginal_update_rule` / `@define_average_energy`
      — `rules:spike`, `rules:specs`, `rules:malformed`
- [x] `RuleSpec`/`NodeSpec` registry, per-module const + discovery (never `push!` into a
      shared global — precompilation hazard, see `PLAN.md`) — `__message_passing_registry__`
      is a `const` created in the defining module by `@define_registry`, and `registries()`
      discovers them across loaded modules (`registry.jl`); `registry:in-process`,
      `registry:lifecycle`. *(Implemented in Phase 3; the box was left unticked until the
      post-Phase-4 audit.)*
- [x] dependency language with the four selectors + static-arity enforcement — plus custom
      selectors, consumed and scored declared separately (#9), and definition-time checks
      of targets, groups, joint order and partition coverage (`dependencies:*`)
- [x] `RuleContext`, `buffer_like`, and the `preallocate` keyword (**not** `@allocate` —
      the in-body macros are deleted, see Phase 0's rule-syntax entry) — `inplace:*`: `rule`
      and `rule!` agree, and the kernel with a provided buffer allocates 0 bytes. The
      writable-output trait that falls back to allocating is engine behaviour, Phase 7
- [x] registry-backed errors; `check_rules()`, `check_rule_ambiguities()` — the error tells
      *no rule of this shape* from *type mismatch* and lists near misses slot by slot; both
      checkers take modules to scope them (`diagnostics:*`)
- [x] argument containers: sorted single keys and type-level joint keys (`Val((:y, :x))`),
      measured `@inferred` and allocation-free on 1.10 with a negative control. **No
      symbol is formed at run time.** `gate:containers*` (tagged `:alloc`); on 1.10.12
      `args.q[:y, :x]` compiles to `getfield(q.joints, 2)`, see `DISCUSSION.md` §3.16
- [x] interactive surface, in full: `@call_message_update_rule`/`@call_marginal_update_rule`/
      `@call_average_energy` and their `@which_*` counterparts (+ function forms, all named after
      the definition macros they mirror), `list_rules(node[, edge]; algorithm)` (renamed from
      `rules`, per the naming rule), rule source on display, `rule_coverage`, the coverage matrix,
      `text/plain` and `text/html` display, and a visualisation entry point, `visualize_spec`,
      which with no backend loaded is a `MethodError` carrying a "load X to enable" hint
- [x] CI assertion: `ExponentialFamily` absent from the dependency closure — `quality:closure`,
      checked on the resolved graph *and* on what a fresh process loads
- [x] **resolve open items #3, #9, #10, #11, #12** — signed off 2026-09-22, see the entry
      brief and `PLAN.md` § Open items. Body slots are now `(output, algo, ctx, args, ann)`
- [x] **#13 (approximations protocol)** — parked by the user until the late phases; the
      `linalg` service stays documented unstable, which does not hold Phase 3 open
- [x] missing-input path semantics confirmed — exactly as v6, in the `execute_rule` docstring
- [x] **registry lifecycle test matrix**: fresh-process load after precompilation, both
      extension load orders, definitions in nested modules, supported interactive
      redefinition. Test duplicate signatures separately from ambiguous ones —
      `registry:lifecycle` (`:slow`, fixture packages under `test/fixtures/`, each probe a
      fresh process) and `registry:in-process`. Ambiguity is step 8's `check_rule_ambiguities`
- [x] purity and RNG ownership contracts specified, including permitted output/scratch
      writes and the distinction between the audit policy and differentiation support — the
      `ispure` and `RuleContext` docstrings
- [x] built test-first throughout — each step's tests were written ahead of its
      implementation and landed in the same commit
- [x] `lib/MessagePassingRulesBase/test/` with its own `runtests.jl`, and **a CI job running it
      on 1.10, 1.11 and 1.12** (`LibTests.yml`, the same matrix as `ci.yml`). The
      `ExponentialFamily`-absent assertion above lives in that job

**Decision checkpoints:** #2 (keep generalization deferred unless needed), #3 (factorisation
keying), and #9–#13 (resolve before the API freezes).

---

## Phase 4 — `MessagePassingRulesTestUtils`

**Goal:** test tooling as its own package, consumed via `[extras]`.

**Exit criteria**
- [x] `@test_rules` successor: numerical output, type promotion (default on),
      `rule`/`rule!` agreement, optional non-allocating flag — `@test_message_update_rule`,
      `@test_marginal_update_rule`, `@test_average_energy`; failures point at the table's line
- [x] **node-definition verification** — reference update computed from `nodefunction`
      for a bounded initial subset, with separate shape and scale assertions —
      `@verify_message_update_rule`: BP and naive VMP, point masses substituted, finite
      discrete inputs enumerated, ≤2 continuous inputs integrated (HCubature). Scale is
      checked only when the shape holds
- [x] registry-backed coverage check: every `RuleSpec`/`NodeSpec` has a test; record the
      actual selected rule so a fallback cannot conceal an untested specialization —
      `check_rule_coverage(modules...)`; a broader rule answering for a specific one stays
      reported (`coverage:selected-rule`)
- [x] **migration checker**: runs a v6 and a v7 rule on identical inputs and asserts they
      agree — the tool that makes downstream (and agent-driven) migration verifiable —
      `compare_with_reference` and version-stamped fixtures in TestUtils; `V6Oracle.jl` and
      `check.jl` in `compat/v6-comparison`, run by the `v6-comparison` CI job on 1.10
- [x] disagreements with v6 investigated and recorded as migration bugs or deliberate
      mathematical corrections; analytic/finite-difference derivative checks cover both
      allocating and in-place paths — the mechanism is built (`DeclaredDisagreement`, pinned
      `KNOWN_V6_FINDINGS`) and its first real case investigated: v6's variational
      `NormalMeanVariance` rules are wrong for non-point-mass `q_v` (ReactiveMP.jl#669), a
      `:correction` for Phase 5. Investigating each ported rule is Phase 5's work, per rule.
      Derivative checks: `@test_rule_derivatives`, through `rule` and `rule!`

Definition verification lands **before** Phase 5, not after: it is the difference between
checking ported rules against v6's output and checking them against the mathematics. Expect
it to surface rules that were already wrong.

---

## Phase 4.5 — Engine design and first cut

**Goal:** the first cut of the real v7 engine, written directly in `src/` and built for four
slice cases, so that the rule/engine *interface* is proven against the engine that will ship
before hundreds of rules are ported against it.

Rule kernels and test utilities can be developed without an engine. That does not
establish that their interface with the engine is correct — **this is the single largest
planning risk**, and the cheapest insurance is a small end-to-end proof first.

**Restructured on 2026-09-23** (`DISCUSSION.md` §3.18). An earlier version of this phase
asked whether the slice should be a bridge that lets v6's `MessageMapping` call base-package
rules. It is not: most of the slice runs through the code Phase 7 was to demolish — the
mixtures' `activate!`/`ManyOf`/`reverse` wiring and the delta layouts' own rule path — and
v6 and v7 cannot share a process, so the free-energy comparison is fixture-based either way.
Decided instead:

- **Phase 4.5 absorbs the start of Phase 7.** One engine design session, then the real
  engine for the slice's nodes. Phase 7 becomes *complete the engine*.
- **v6 is a fixture source only.** Fixtures are recorded from `compat/v6-comparison`
  **before anything is deleted**. The v6 engine is deleted when the new one lands; v6 rules
  and their tests are deleted per directory as Phase 5 ports them. `src/` never holds two
  engines, and the suite shrinks rather than going red.
- **ReactiveMP takes a hard `[deps]` entry on `MessagePassingRulesBase`** — `[sources]` on
  1.11+, developed at test time on 1.10, as TestUtils does. `ci.yml`, `make test` and
  `make docs` gain the develop step.
- **Downstream breakage is accepted until the release.** Only a small internal group uses the
  branch, and it is verified locally; `IntegrationTest.yml` is not a gate before Phase 8.

**Status: not started. Next is the design session**, before any code.

### Contracts already made — the engine implements them, it does not revisit them

Decisions are in `PLAN.md` § Open items, reasons in `DISCUSSION.md` §3.16–3.17. Each was
re-verified against `lib/MessagePassingRulesBase` in the post-Phase-4 audit.

- **Rules see no envelope.** `Message`/`Marginal` stay in the engine; the engine unwraps
  them and calls a rule with raw distributions in `RuleArgs` (`args.m`, `args.q`), sorted
  single keys and type-level cluster keys (`q[:y, :x]`), built with
  `MessagePassingRulesBase.RuleArgs`/`Messages`/`Marginals`.
- **Groups arrive full length.** A group input is a tuple in member order with `nothing`
  where the dependency's selection leaves a member out; an empty selection is `()` and is
  already satisfied — it must never stall a stream combination.
- **Annotations are two-way.** The engine builds `RuleAnnotations(m = …, q = …, out = …)`:
  the annotations that arrived with each input, keyed like the inputs, plus the sink the rule
  writes. v6's `AnnotationDict` and its post-rule processors map onto `out`.
- **The context carries the node.** `RuleContext(node, product, linalg, rng)`, built per
  node/edge; `product` replaces the throwaway `randomvar` in v6's `rules/mixture/switch.jl:11`;
  `linalg` stays unstable (#13 parked).
- **Missing inputs as v6**: no rule call, no post-rule processors, result `missing`
  (`execute_rule` docstring, `rulespec.jl:113-116`).
- **Resolution before execution.** `find_message_rule`/`find_marginal_rule`/
  `find_average_energy` return a `RuleSpec` or `RuleNotFound` and never throw; the engine's
  fallback sits on the `RuleNotFound` branch; `execute_rule(spec, output, algorithm, ctx,
  args, ann, target)` — seven arguments, the target included (`rulespec.jl:107`) — never
  catches.
- **Dependencies are declared per algorithm** (`DependenciesSpec`, `dependencies_spec(node,
  algorithm)`); `nothing` means the engine's default scheme (own cluster → messages minus
  self, other clusters → marginals). Consumed inputs and the free-energy partition are
  separate (#9). `static_inputs = :fold` is the delta static-gating policy; a singleton
  cluster's marginal *is* the variable's marginal (`q_out` aliasing, engine invariant).
- **Buffers are engine internals** (#10): reuse is unspecified; outsiders copy; getters copy
  by default; `InputArgumentsAnnotations` deep-copies.

### What v6 does, and where — to replace, and to record fixtures from

All re-verified in the post-Phase-4 audit.

- rule call: `MessageMapping`, `src/message.jl:570-738`. The callable is at `:657`; it builds
  `ruleargs` at `:692-703` and calls `rule(ruleargs...)` at `:704`; the fallback runs on a
  `RuleMethodError` at `:707-712` (`NodeFunctionRuleFallback`, `src/rules/fallbacks.jl:60-73`);
  pre-/post-rule annotation processors at `:678-682`/`:722-728`, the missing-input
  short-circuit at `:685-690`. `DeferredMessage` computes lazily and caches, `:447-494`;
- marginal rules: `MarginalMapping`, `src/marginal.jl:251-322`;
- average energy and free energy per node: `src/score/node.jl` (`score` at `:7`, `:26`, `:80`);
- dependency wiring: `activate!` in `src/nodes/dependencies.jl:59` and
  `src/nodes/nodes.jl:327`; clusters in `src/nodes/clusters.jl:129`;
- the mixture `reverse(...)` (#6): `normal_mixture.jl:176,183`, `gamma_mixture.jl:166,173`,
  inside `collect_latest_marginals`. It orders only the `combineLatest` trigger, so what is
  to be recorded is **emission order**, not values.

**Gaps in today's tests that the fixtures and the new engine's tests must close.** The mixture
`reverse` order is not pinned (`normal_mixture_tests.jl:357-397` checks names only). The
missing-input path is pinned only under `LogScaleAnnotations`, and without asserting that the
rule was not called (`test/annotations/logscale_tests.jl:142-220`). Retained-message
immutability is not pinned at all, and v6's `Message` holds an `AnnotationDict` that can be
mutated in place.

### Step 0 — record the v6 fixtures, before anything is deleted

`compat/v6-comparison` gains an RxInfer version compatible with ReactiveMP 6.5.0, and
records, for each slice model: free-energy trajectories, posteriors, per-message log scales,
emission order (the mixture `reverse` order included) and the missing-input behaviour.
`MigrationRecord` is rule-shaped, so this needs a trajectory-shaped fixture alongside it.

### The design session's agenda

Carried over from the earlier planning questions:
1. **Which four slice cases** — one each for BP, structured VMP, a mixture (variadic group)
   and a delta node, and whether the delta one needs `static_inputs = :fold` in the first cut.
   Proposed: `NormalMeanVariance` BP (its rules are already ported in `check.jl`),
   `NormalMeanPrecision` VMP, the `NormalMixture((:m, k))` canary, and delta + `Unscented`
   without `:fold`, static gating as a stretch (Phase 0 has fixtures for it).
2. **The fixture format and source** — Step 0 above; proposed as stated there.
3. **How annotations and log scales are compared**: `Message ==` ignores annotations, so the
   check must be explicit (`PLAN.md` § Verification, annotation gate). Proposed: per-message
   log scales recorded through callbacks and compared with a tolerance.

Moved up from Phase 7:
4. dependency-to-stream wiring for groups (`__collect_latest_updates` must collapse
   consecutive same-name interfaces), and with it the mixture `activate!` demolition;
5. the `Message`/`DeferredMessage` envelope (immutable `Message`, annotations as a type
   parameter);
6. preserving edge order when building clusters — GraphPPL's factorisation indexes the
   original flat list, the single biggest correctness trap — and plumbing `EdgeLabel.index`
   through RxInfer rather than re-deriving group indices from neighbour position (#7);
7. buffer-reuse eligibility for the engine's own retainers (#10);
8. where `RuleContext` is built and held, and where the fallback lives.

### Exit criteria
- [ ] v6 fixtures recorded for the slice models (Step 0), before any v6 code is deleted
- [ ] a working end-to-end inference in the new engine over the slice: ordinary belief
      propagation, structured VMP, a mixture (variadic group), and a delta node
- [ ] free energy agrees with the recorded v6 trajectories on the same models
- [ ] annotations and log scales agree with the recorded ones, compared explicitly (see the
      annotation gate in `PLAN.md`)
- [ ] a retained-value test: hold a message across several updates and confirm it is not
      mutated underneath you
- [ ] the missing-input path pinned by a test (rule not called, post-rule processors skipped)
- [ ] mixture emission order agrees with the recorded v6 order (#6)
- [ ] edge order and group indices preserved through integration (#7)
- [ ] the v6 engine deleted; ReactiveMP depends on `MessagePassingRulesBase`

Do not start Phase 5 until this passes.

---

## Phase 5 — `StandardMessagePassingRules`

**Goal:** standard distribution nodes plus arithmetic (`+`, `-`, `*`, dot).

**Exit criteria**
- [ ] JuliaSyntax-based migration tool, run with ReactiveMP v6 loaded as an oracle for
      `interfaces(fform)` (do **not** regex-guess on `_`)
- [ ] migrated per rule directory, diffs reviewed per directory; each v6 directory and its
      tests are deleted in the same commit that ports it (v6 fixtures are recorded first,
      see Phase 4.5 Step 0)
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
      for the diagnostics. Per #11, that is the host's `is_delta_node_compatible` guard
      carried over, with the error extended to name the alternative method
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

## Phase 7 — Complete the engine

Phase 4.5 now builds the engine's first cut and settles its design (restructured
2026-09-23, `DISCUSSION.md` §3.18); what remains here is completing it once Phases 5–6 have
ported the rules. The items that moved to Phase 4.5 are group stream wiring, the `Message`
envelope, edge order, `EdgeLabel.index` (#7) and the mixture `reverse` (#6).

Known scope:
- [ ] every node supported, as Phases 5–6 port their rules — including the remaining
      mixture `activate!` demolition via variadic groups (~70–85% deletable)
- [ ] `EqualityChain` `BitVector` → `Vector{Bool}`
- [ ] engine diagnostics: `check_everything_pure`, `check_everything_inplace`, checked buffers
- [ ] RxInfer plumbing for the new engine
- [ ] explicit checks on scheduling order, annotations, retained values and free energy —
      not just numerical rule equality — for every ported node, against recorded v6 fixtures

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

Tracked with stable numbers in `PLAN.md` § Open items. Of 14 items, #1, #3, #8, #9, #10,
#11, #12 and #14 are resolved (#3 and #9–#12 at the Phase 3 sign-off), and #4 is **deferred by
decision** (Phase 0; reopened only by a concrete ruleset use case). #13 is **parked** by the
user and holds back only the `linalg` context service. #6 and #7 are engine integration
requirements, closed in Phase 4.5. #2 remains deferred unless needed. #5
(Reactant/StableCholesky) belongs to a separate effort and does not block this rewrite.

## Structural note

Rule kernels and test utilities can be developed independently of the engine. However,
**Phase 5 bulk migration is gated on Phase 4.5**, and release is gated on full engine and
downstream integration. Engine independence does not establish interface correctness.
Since the restructuring (`DISCUSSION.md` §3.18) the engine Phase 4.5 builds is the real one,
not a proof alongside v6, so Phase 5 ports rules straight into it and Phase 7 completes it.
