# Phases — rule/node system rewrite

State tracker for the work described in `PLAN.md` (decisions) and `DISCUSSION.md`
(rationale). This file is the **volatile** one: it says where we are, not what we decided.

**Rule: update this file in the same commit as the change it describes.** Status claimed
without a diff alongside it is how a tracking file starts lying.

Branch: `refactor/rule-node-system-rewrite`

---

## Next action

**Phase 0** — the throwaway spike, in `spike/`. Phase P is complete and was re-verified
against the working tree, not just against this file: the disposition inventory is assigned
and CI-gated, the repository layout and Julia floor are decided, and the comparison
environment resolves. Phase 0 answers the questions that cannot be walked back.

The spike is committed as it is built, so its measurements are reproducible, and **deleted
in the commit that closes Phase 0** — the findings survive in `DISCUSSION.md`. Measurements
run on the **1.10 floor** and on 1.13; a gate that passes only on the newest Julia says
nothing about the version this package supports.

---

## Status at a glance

| # | Phase | Status |
|---|---|---|
| — | Initial design documented in `PLAN.md`, `DISCUSSION.md` | **done; open decisions below** |
| — | External design review; contradictions reconciled | **done** |
| P | Prep: disposition inventory, layout and environment decisions | **done** |
| 0 | Spike: dispatch gate, syntax samples, delta dependency semantics | **done — 13/13** |
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
        a `make test` fast subset that drops `:quality` would silently un-gate the inventory
- [x] **repository layout decided: monorepo under `lib/`, split at Phase 6.** Boundaries
      are still moving (#9–#13 are unresolved API decisions), and a cross-package change is
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
        Manifest. Documented in `lib/README.md`

---

## Phase 0 — Spike (throwaway)

**Goal:** test dispatch and dependency assumptions before committing to the API.

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
      `(output, algo, ctx, args, ann, node)` in canonical order, dispatch carried by the
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
      confirmed: `mixture.jl:142` takes three positional arguments while its only caller
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
      `args.ann_in[:out]` — kept out of dispatch, since an annotation must not select the
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
- [ ] Aqua `ambiguities` enabled. **Measured in Phase P: 322 ambiguous pairs**
      (`Aqua.detect_ambiguities(ReactiveMP; recursive = true)`), which split into five
      unrelated problems that should be budgeted and fixed independently:

      | count | source | character |
      |---|---|---|
      | 253 | `src/helpers/algebra/{permutation_matrix,standard_basis_vector,companion_matrix}.jl` | custom array types declaring `*`/`dot` against bare `AbstractMatrix`/`AbstractVector`, colliding with `ArrayLayouts`, `PDMats`, `FillArrays` and `LinearAlgebra`. Cleanup is independent of rule dispatch; the inventory moves the used helpers with Flow/AR and deletes CompanionMatrix |
      | 27 | `rule`/`marginalrule` dispatch | **one single shape**, repeated: the delta catch-all `rule(::F<:Function, …, meta::DeltaMeta, …, node::DeltaFnNode)` (`delta.jl:78`, `:104`) against the `meta::Any` arithmetic rules — `src/rules/addition/in2.jl:1`, `src/rules/subtraction/{out,in1,in2}.jl:1` and `src/rules/multiplication/marginals.jl:31`. Neither is more specific — delta wins on `meta`/`node`, the arithmetic rule wins on `fform`/`on`/`messages`. This is the only category the new dispatch design is claiming to eliminate, and it is a useful Phase 0 target |
      | 23 | `src/fixes.jl` | the deliberate upstream hot-fixes; expected to disappear when upstream releases |
      | 11 | `nodes/predefined/uninformative.jl` | `prod` for `Uninformative` against `BayesBase`'s `PreserveTypeProd` methods; separate from the two known piracies in `uniform.jl` |
      | 8 | scattered | `gcv.jl`, `cvi.jl`, `message.jl`/`marginal.jl`, `nodes.jl` vs the mixtures |

      **Read ambiguity reports with care.** An earlier version of this table cited the
      arithmetic rules as `rule.jl:358`/`:392`. Those two lines are the
      `function ReactiveMP.rule(` and `function ReactiveMP.marginalrule(` headers *inside the
      macro's `quote` block*, so **every one of the ~490 generated rule methods reports that
      same source location**. `Method.file`/`Method.line` therefore cannot identify a rule
      today; v6 recovers the node and interface names from the *signature* instead
      (`get_node_from_rule_method`, `src/rule.jl:1664-1690`). That is an argument for the
      registry independent of the ambiguity count.

      Originally reported on Julia 1.13.0 against a local root `Manifest.toml` (gitignored,
      not committed). No machine-readable report or exact resolution was retained. The count is both
      Julia-version and resolution dependent, so re-measure before acting rather than
      treating 322 as fixed. Zero pairs were reported to have neither side in ReactiveMP.
      Per-type counts in inventory notes may overlap and must not be added as disjoint totals
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
- [ ] `RuleContext`, `buffer_like`, and the `preallocate` keyword (**not** `@allocate` —
      the in-body macros are deleted, see Phase 0's rule-syntax entry)
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

Tracked with stable numbers in `PLAN.md` § Open items. Of 14 items, #1, #8 and #14 are
resolved; the remaining 11 are not all immediate blockers. #3 and #9–#13 must be settled
before Phase 3's API freezes. #6 and #7 are
engine integration requirements. #2 remains deferred unless needed; #4 may be deferred
pending a concrete ruleset use case. #5 (Reactant/StableCholesky) belongs to a separate
effort and does not block this rewrite.

## Structural note

Rule kernels and test utilities can be developed independently of the engine. However,
**Phase 5 bulk migration is gated on Phase 4.5**, and release is gated on full engine and
downstream integration. Engine independence does not establish interface correctness.
