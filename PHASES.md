# Phases — rule/node system rewrite

State tracker for the work described in `PLAN.md` (decisions) and `DISCUSSION.md`
(rationale). This file is the **volatile** one: it says where we are, not what we decided.

**Rule: update this file in the same commit as the change it describes.** Status claimed
without a diff alongside it is how a tracking file starts lying.

Branch: `refactor/rule-node-system-rewrite`

`file:line` citations in the three documents were re-verified against the code after Phase 4.5
closed (`0b8f1caa`) and not since; the Phase 5 briefs cite the code as it was when each was
written. Citations into v6 code name ReactiveMP 6.5.0's layout. Only the unported nodes, their
rules and the helpers and approximations they use are still in `legacy/v6/`, under the same
paths; everything else of v6 (its engine and rule-system files, `clusters.jl`,
`dependencies.jl`, `score/`, the fallbacks, and every ported node's files) is only in the 6.5.0
release and in git. A later reformat or edit moves them again, so re-check before relying on
one.

---

## Next action

**Phase 6, step 7: BIFM** (§ Phase 6, *Entry brief*). Step 7 is briefed (§ Phase 6, *Step 7
brief*):
- scratch space for rules first, per rule and write-before-read (§3.44);
- then BIFM and BIFMHelper, BIFM stateless, its rules reading their own edge's message;
- MvNormalMeanPrecision's `TerminalProdArgument` marginals move to Standard;
- the free energy of a BIFM model is an explicit error.

Steps 1–6 are done:
- the numerics;
- Delta;
- GaussianCoupling, Probit and GCV;
- the autoregressive family;
- ContinuousTransition;
- the Pólya nodes, whose rules towards the weights read their own edge's message (§3.41), with
  both energies corrected and BinomialPolya's by Gauss–Hermite (§3.43).

Each is in its own package, compared with v6 and covered by an engine fixture.

**Everything not done yet, and where it is recorded**, so nothing is lost between sessions:

| What | Where it lands | Recorded in |
|---|---|---|
| a joint cluster holding some members of a group with other interfaces (`activate!` refuses it) | when a node needs one | § *Cases (b)–(d)*, case (d) |
| typed annotations (`Message{D, A}`), with the log-scale milestone | Phase 7, after the migration | brief item 3; `DISCUSSION.md` §3.23 |
| Aqua's `ambiguities` check re-measured and re-enabled | Phase 7 | § Phase 7 |
| the `.github/` workflows brought up to date (1.13, the step-4 layout) before the first PR | Phase 7 | § Phase 7 |
| RxInfer adapted to the new engine API | Phase 7 | § Phase 7 |
| `LogScaleAnnotations`' all-point-mass fallback does not look inside a `FactorizedCluster` | the log-scale milestone, Phase 7 | Phase 5 review |
| `Uninformative × missing` is `missing` through `UninformativeProd` and `Uninformative()` through `GenericProd` | with the upstream BayesBase identity item | Phase 5 review |
| BayesBase owns `Uninformative` as a product identity, as it treats `missing`, and the Uniform(0, 1)×Beta product moves upstream; Standard's `UninformativeProd` and the Uniform piracy then go | upstream, a non-breaking BayesBase (or ExponentialFamily) release | § Phase 5, step 3 |
| ExponentialFamily 2.6's `mean(logdet, ::InverseWishart{Float32})` is a Float64 (`d * log(2)`), so MvNormalMeanCovariance's energy with an InverseWishart `q_Σ` is too (`@test_broken` in Standard), and its `mean(cholinv, ::InverseWishart{BigFloat})` fails (InverseWishart's energy table runs in Float64 only), and its `mean(loggamma, ::GammaShapeRate)` is a Float64 (GammaMixture's switch and energy tables run in Float64 only) | upstream, an ExponentialFamily patch release | ExponentialFamily.jl#322 |
| `public_equivalent` owned by BayesBase and extended by ExponentialFamily for its Fast types; the base package's copy then goes | Phase 8, the ecosystem integration | `DISCUSSION.md` §3.29 |
| the RNG as an activation option (the engine passes `Random.default_rng()` until then), and `*`'s number of samples (3000, v6's) configurable | Phase 7 | `DISCUSSION.md` §3.32 |
| `*`'s sampled messages are unnormalised sums, as in v6: a missing constant in their log-scale | the log-scale milestone, Phase 7 | § Phase 5, *Step 7 brief* |
| `@test_message_update_rule` cases taking incoming annotations (`ann.m`), for rules that read log scales | when a second node needs it | § Phase 5, *Step 8 brief* |
| the Delta package's missing `LibTests` job | Phase 7, with the workflows | § Phase 7 |
| log scales: fix v6's gaps or drop the feature (and with it Mixture's rules) | the log-scale milestone, Phase 7 | `DISCUSSION.md` §3.37 |
| the engine calls `missing_services` when it resolves a rule, so a declared service that is `nothing` is an error there rather than inside the rule | Phase 7 | § Phase 5, *Step 8 brief* |
| user rule sets beyond one-level extensions | not planned; #4 | `DISCUSSION.md` §3.23 |

The rule registry was clarified with the user after step 4 and **stays as it is**: lookup is
the base package's method table, global already, and the per-module registries are
introspection only (`DISCUSSION.md` §3.23, Correction 25).

Phase 4.5's four slice cases and its ground rules are in § Phase 4.5: the step-4 clean cut,
work on **Julia 1.13 only** wired with `[sources]`, and **no CI runs yet**, everything verified
locally (`DISCUSSION.md` §3.22).

Phases 0–5 are closed. `lib/MessagePassingRulesBase` is the rule system;
`lib/MessagePassingRulesTestUtils` is its test tooling; `lib/StandardMessagePassingRules` holds
every standard node (the distributions, arithmetic, logic and the mixtures);
`lib/MessagePassingRulesApproximations` holds `Unscented` and `smoothRTS`;
`lib/DeltaMessagePassingRules` holds the Delta node; and `compat/v6-comparison` holds the v6
oracle, the comparisons and the engine fixtures. ReactiveMP itself is the engine on the new rule
system; the nodes Phase 6 ports are in `legacy/v6/`. The
tooling's first finding was a real v6 bug: the variational `NormalMeanVariance` rules use
`E[v]` instead of `1/E[1/v]` for a non-point-mass `q_v` (ReactiveMP.jl#669). It was **corrected
when NMV was ported in step 3**, and the comparison declares it.

**Picking this up on another machine.** Everything lives in the repository. With Julia 1.13:

```bash
git switch refactor/rule-node-system-rewrite && git pull
make test test-base test-testutils test-standard test-approximations test-delta test-gaussian-coupling test-probit test-gcv
julia --startup-file=no --project=compat/v6-comparison -e 'using Pkg; Pkg.instantiate()'
for s in check compare_standard compare_approximations compare_delta compare_gaussian_coupling compare_probit compare_gcv; do
    julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/$s.jl
done
julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/record_engine_fixtures.jl --check
```

Then read, in order: `CLAUDE.md`, `PLAN.md`, `DISCUSSION.md` §4 *Corrections* and §3.14–3.37,
and this file's § Phase 5 (its step briefs are the pattern Phase 6 follows) and § Phase 6. Working conventions: one commit per step, failing test first,
`PHASES.md` and `CHANGELOG.md` updated in the same commit, descriptive names rather than
generic ones, and no comments that only narrate.

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
| 4.5 | **Engine design and first cut** — the engine refactored in place for four slice cases *(absorbs the start of 7)* | **done**: steps 0–4, the algorithm reconciliation and all four slice cases |
| 5 | `StandardMessagePassingRules` | **done**: steps 1–9, and the post-close review's findings resolved |
| 6 | `MessagePassingRulesApproximations` + node packages | entry brief written; steps 1 (numerics), 2 (Delta), 3 (GaussianCoupling, Probit, GCV), 4 (AR, ConjugateAR, SoftDot), 5 (ContinuousTransition) and 6 (the Pólya nodes) done; step 7, BIFM, next |
| 7 | Complete the engine — diagnostics, RxInfer adaptation, what the slice did not need | not started |
| C | Cleanup: the repository rid of historical remarks, before the release | not started |
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
        `base` 43, `engine` 32 *(35 and 40 since `Message`, `Marginal` and six accessors were
        corrected to `engine` after step 4, as `PLAN.md` § Package split decided)*, `delete` 21
        *(22, and `standard` 57, since `NormalMixtureNode` became a deletion in Phase 4.5)*, `node:Flow` 16, `node:Delta` 11,
        `models` 11, `node:Polya` 10, `node:Autoregressive` 9, `approximations` 7,
        `node:ContinuousTransition` 5, `node:BIFM` 5, `node:DiscreteTransition` 3.
        `StandardMessagePassingRules` is distributions, arithmetic, logic and mixtures;
        domain-specific models (GCV, Probit, SoftDot, GaussianCoupling) go to a separate
        package **that is not yet named**, so the token is `models` *(since Phase 5 step 6,
        each gets its own node package, `DISCUSSION.md` §3.30)*
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
      Pay the split cost once, at a known gate. See `PLAN.md` § Repository layout. *(The gate
      moved to Phase 8, with registration, in the Phase 6 entry brief; `DISCUSSION.md` §3.40.)*
- [x] **Julia floor decided: stays at 1.10.** Nothing in the design requires more. The
      `ScopedValue` in an earlier draft of § Dispatch axes was never necessary — the
      context is an ordinary object passed into the rules, most likely held by
      `MessageMapping`, and a plain default argument gives the same behaviour.
      *(Superseded in Phase 4.5, user: work targets Julia 1.13 only, wired with `[sources]`;
      the floor and the 1.10 workarounds are revisited when the packages are registered.
      See `DISCUSSION.md` §3.22.)*
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
      rather than a resurfacing. *(Since the algorithm reconciliation, a
      `DefaultAlgorithmExtension` gives a one-level overlay without one.)*
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
      throughout (`target = :out`, `m[:μ]`, `interfaces = [:out, ...]`), group members
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
      target, but an indexed target `target = (:m, k)` has to bind `k`, which is a runtime
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
      tuple rather than of the new design — `ManyOf{N,T}` had it — and it is the price
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
      | 71 | `src/helpers/algebra/companion_matrix.jl` | same shape — and `CompanionMatrix` has **zero references anywhere in `src/` or `test/`**, so this is entirely dead weight *(wrong: AR builds it with `as_companion_matrix`; Phase 6 entry brief)* |
      | 27 / 25 | `delta.jl` / `rule.jl` | **one** repeated shape: the delta catch-all against the `meta::Any` arithmetic rules. The only category the new dispatch design claims to eliminate. **27 distinct pairs, not 52** — 25 of them touch both files. And `rule.jl` here is the `@marginalrule` *template* (`rule.jl:406`), where every generated method reports its location, so it names the arithmetic rules in `src/rules/{addition,subtraction,multiplication}/`, not code in `rule.jl` (see `DISCUSSION.md` §5) |
      | 23 | `src/fixes.jl` | deliberate upstream hot-fixes; they leave when upstream releases |
      | 11 | `nodes/predefined/uninformative.jl` | `prod` for `Uninformative` against BayesBase's `PreserveTypeProd` |

      (A pair is counted against both files it touches, so the column does not sum to 322.)

      **Why not now.** Three files account for the large majority, `INVENTORY.md` already
      sends all three out of this package with Flow/AR and marks `CompanionMatrix` for
      deletion *(it goes to AR since the Phase 6 entry brief)*, and the 27 rule-dispatch pairs are what the rewrite removes by construction.
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
  keeps the one-way door closed without building anything. *(Refined at the algorithm
  reconciliation: the default scheme already follows the factorisation; only a node that
  ignores it declares its own algorithm.)*

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
      `:correction` for Phase 5 *(corrected earlier, when Phase 4.5 step 3 ported NMV)*.
      Investigating each ported rule is Phase 5's work, per rule.
      Derivative checks: `@test_rule_derivatives`, through `rule` and `rule!`

Definition verification lands **before** Phase 5, not after: it is the difference between
checking ported rules against v6's output and checking them against the mathematics. Expect
it to surface rules that were already wrong.

---

## Phase 4.5 — Engine design and first cut

**Goal:** the v7 engine's first cut, refactored in place in `src/` and built for four slice
cases, so that the rule/engine *interface* is proven against the engine that will ship before
hundreds of rules are ported against it.

**What v7 changes in the engine, and what it keeps** (user, 2026-09-23). The underlying
reactive machinery is kept: Rocket streams, variables, the equality chain, message products,
deferred messages and scores. What is replaced is how rules are found, fetched and called, and
how nodes and rules are defined and created. That covers the per-node activation overrides,
the positional edge handling, and the rule-call path in `MessageMapping`/`MarginalMapping`.
It is an improvement of the rule-call behaviour plus a clean-up, not a new engine. RxInfer
needs adjusting where node and rule creation changes, and little elsewhere.

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
  **before anything is deleted**. The v6 rule-call path, node-creation path and per-node
  activation are replaced in place, with no second copy kept alongside. `src/` never holds
  two engines, and the suite shrinks rather than going red. *(Sharpened by the clean cut,
  §3.22: step 4 moves every unported node, and the whole v6 rule system, to `legacy/v6/`
  at once, rather than Phase 5 deleting them directory by directory.)*
- **ReactiveMP takes a hard `[deps]` entry on `MessagePassingRulesBase`**, wired with
  `[sources]`, on Julia 1.13 *(the 1.10 develop-at-test-time steps are dropped, §3.22)*.
- **Downstream breakage is accepted until the release.** Only a small internal group uses the
  branch, and it is verified locally; `IntegrationTest.yml` is not a gate before Phase 8.

**Status: signed off 2026-09-23. Steps 0 (fixtures), 1 (the design session), 2 (base
additions), 3 (rule ports) and 4 (the engine core, case (a)) are done, as is the algorithm
reconciliation. Cases (b), VMP, (c), the mixture, and (d), Delta, are done: Phase 4.5 is
closed.**

### Contracts already made — the engine implements them, it does not revisit them

Decisions are in `PLAN.md` § Open items, reasons in `DISCUSSION.md` §3.16–3.17. Each was
re-verified against `lib/MessagePassingRulesBase` in the post-Phase-4 audit.

- **Rules see no envelope.** `Message`/`Marginal` stay in the engine; the engine unwraps
  them and calls a rule with raw distributions in `RuleArgs` (`args.m`, `args.q`), sorted
  single keys and type-level cluster keys (`q[:y, :x]`), built with
  `MessagePassingRulesBase.RuleArgs`/`Messages`/`Marginals`.
- **Groups arrive full length.** A group input is a tuple in member order with `nothing`
  where the dependency's selection leaves a member out; an empty selection takes no stream,
  so it never stalls a combination, and reaches the rule as a tuple of `nothing`s,
  `(nothing,)` for one member (`EmptyGroup`, case (d)).
- **Annotations are two-way.** The engine builds `RuleAnnotations(m = …, q = …, out = …)`:
  the annotations that arrived with each input, keyed like the inputs, plus the sink the rule
  writes. v6's `AnnotationDict` and its post-rule processors map onto `out`.
- **The context carries the node.** `RuleContext(node, product, linalg, rng)`, built per
  node/edge; `product` replaces the throwaway `randomvar` in v6's `rules/mixture/switch.jl:11`;
  `linalg` stays unstable (#13 parked).
- **Missing inputs as v6**: no rule call, no post-rule processors, result `missing`
  (`execute_rule` docstring, `rulespec.jl:136-139`).
- **Resolution before execution.** `find_message_rule`/`find_marginal_rule`/
  `find_average_energy` return a `RuleSpec` or `RuleNotFound` and never throw; an engine
  fallback would sit on the `RuleNotFound` branch, and the first cut has none (user fallbacks
  are Phase 7; until then `RuleNotFound` is raised as `RuleNotFoundError`); `execute_rule(spec, output, algorithm, ctx,
  args, ann, target)` — seven arguments, the target included (`rulespec.jl:141`) — never
  catches. The engine passes it `rule_algorithm(spec, algorithm)`, which is `DefaultAlgorithm()`
  when a `DefaultAlgorithmExtension` inherited the rule (algorithm reconciliation).
- **Dependencies are declared per algorithm** (`DependenciesSpec`, `dependencies_spec(node,
  algorithm)`); `nothing` means the engine's default scheme (own cluster → messages minus
  self, other clusters → marginals). Under `DefaultAlgorithm` that is the usual case, and
  the scheme follows the factorisation; an extension without its own declaration inherits
  the default's. Consumed inputs and the free-energy partition are
  separate (#9). `static_inputs = :fold` is the delta static-gating policy; a singleton
  cluster's marginal *is* the variable's marginal (`q_out` aliasing, engine invariant).
- **Buffers are engine internals** (#10): reuse is unspecified; outsiders copy; getters copy
  by default; `InputArgumentsAnnotations` deep-copies.

### What v6 does, and where — to replace, and to record fixtures from

All re-verified in the post-Phase-4 audit, **against v6**. Step 4 has since replaced these in
`src/`: the line numbers are ReactiveMP 6.5.0's. The files were kept under the same paths in
`legacy/v6/` until Phase 5 step 9 deleted them; they are in the 6.5.0 release and in git.

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
  inside `collect_latest_marginals`. It orders only the `combineLatest` trigger. *(That this
  leaves values alone turned out half wrong: the trigger order is the VMP update schedule, and
  its group order changes the trajectory; case (c), §3.24.)*

**Gaps in today's tests that the fixtures and the new engine's tests must close.** The mixture
`reverse` order is not pinned (`normal_mixture_tests.jl:357-397` checks names only). The
missing-input path is pinned only under `LogScaleAnnotations`, and without asserting that the
rule was not called (`test/annotations/logscale_tests.jl:142-220`). Retained-message
immutability is not pinned at all, and v6's `Message` holds an `AnnotationDict` that can be
mutated in place.

### Step 0 — record the v6 fixtures, before anything is deleted

**Done.** `compat/v6-comparison` pins RxInfer 5.5.2, and `record_engine_fixtures.jl` records
seven models *(ten since: `delta_unscented_static` in case (d), `logic_bp` in Phase 5 step
4 and `mixture_bp` in step 8; the script takes model ids to record or check only those)* into `fixtures/engine/<model>.toml`: `bp_iid`, `bp_iid_missing`, `bp_chain`,
`vmp_meanfield`, `vmp_structured`, `normal_mixture` and `delta_unscented`. Each fixture holds
the free energy per iteration, the final posteriors, and every message-rule call **in the
order v6 made it**, with its result and log scale. That order is materialisation order, and it
includes the mixture's. `--check` re-records and compares; it runs locally. The `v6-comparison` CI job still
targets 1.10 and is stale until the workflows are updated (Phase 7).

The fixture is TestUtils' `EngineTrajectory`, holding `RuleCallRecord`s, written as **TOML**
(`save_engine_fixture`/`load_engine_fixture`, compared by `compare_engine_trajectory`).
`Serialization`, which the rule-level `MigrationRecord` uses, only reads back on the Julia
minor that wrote it, and the fixtures were recorded on 1.10.12 and are read on 1.13.

What recording found, all **preserved, not fixed**:
- **Log scales are recorded only for `bp_iid`.** Log scales are a niche feature with known
  gaps; per the user they keep v6's behaviour, and fixing them is **a separate milestone
  after the migration**, with its own plan. Taking easy wins along the way is fine. v6 cannot
  annotate the other models:
  - `NormalMeanVariance(:μ)` with `(m_out::Normal, q_v::PointMass)` sets no `@logscale`
    (`rules/normal_mean_variance/mean.jl:30`), while its mirror `out.jl:40` does;
  - `(:out)` with `(q_μ::Normal, q_v::PointMass)`, which predicting a missing observation
    needs, sets none either;
  - found by reading, and triggered by no model here: the all-point-mass fallback
    (`annotations/logscale.jl:45-49`) handles all-messages or all-marginals, never a mix.
- **A skipped rule call is still traced.** For a missing input, v6 fires the after-rule-call
  event with result `missing` (`bp_iid_missing`). RxInfer treats a missing observation as a
  prediction and refuses free energy with it, so that fixture has none.
- **In `bp_iid`, the posterior's log scale is minus the free energy** (−8.28849), as it
  should be for an exact BP model. It is a cross-check on the recording.
- **`vmp_structured`'s free energy is not monotone.** It reaches 10.41383 at iteration 3 and
  rises to 10.41462 at iteration 5. Recorded as is; not investigated.
- The trace holds message-rule calls only. Marginal-rule calls fire no event through
  RxInfer's callbacks, so their order is not recorded.

### Step 2 — base-package additions

- [x] **a cluster over a whole group.** A cluster is written as the tuple of its members,
      `q[(:y, :x)]`, with `q[:y, :x]` as shorthand. A group's name inside a cluster means all
      of its members jointly: `q[(:in,)]` in `args` and dependencies, `args.q[(:in,)]` in a
      body, and `target = (:in,)` for the marginal rule. `validate_dependencies` and
      `check_rules` accept groups in clusters, keep interface order, and reject a one-member
      cluster of a single interface in favour of `q[:μ]`. Tests: `group-cluster:*`, plus
      the tuple and group joints added to `gate:containers` (inferred, 0 bytes, JET-clean)
- [x] `getnodefn(node, target)` declared and exported, with no methods: `Target(:out)` is
      the forward function with static inputs folded; the engine's node implements it. Test:
      `nodes:getnodefn`. *(The known inverse, first planned as `IndexedTarget(:in, k)`, is the
      algorithm's since case (d), §3.25.)*
- [x] **a marginal rule may return a factorised cluster.** Built as
      `FactorizedCluster((:out, :μ) => q_outμ, (:v,) => q_v)`: a BayesBase `FactorizedJoint`
      of the blocks, labelled with the member tuple each block covers. The labels are
      carried in the type, and `fc[(:out, :μ)]` reads a block. Entropy, `paramfloattype` and
      `convert_paramfloattype` delegate to the joint. `check_factorized_cluster(target, fc)`
      confirms the blocks partition the cluster in its order. Built from literal labels in a
      rule body, it is inferred and allocation-free on 1.10 and 1.13
      (`gate:factorized-cluster`). TestUtils compares it by labels and blocks, and encodes it
      for engine fixtures. Tests: `factorized-cluster:*`, `tables:factorized-cluster`.
      **For Phase 5:** the promotion check rejects a block that passes an input through
      unchanged, as v6's `v = m_v` does, because the output must carry the promoted float
      type of every input. Ported rules must convert such blocks

### Algorithm reconciliation — before step 4

**Found 2026-09-23 (user):** the base package had two built-in algorithms, `BP` and `VMP`, the
standard nodes declared `algorithm = BP` or `VMP`, and tests modelled a structured
factorisation as a `Structured` algorithm. That misreads v6. There is **one** algorithm,
`DefaultAlgorithm`, Bethe free energy minimisation; BP, VMP and structured VMP come from the
factorisation, through the engine's default dependency scheme. `algorithm` replaces `meta`:
it is a rule switcher, or a node's own algorithm where a node needs one. See
`DISCUSSION.md` §3.20.

- [x] base: `BP`/`VMP` removed; `DefaultAlgorithm` is every node's default. Custom
      algorithms come in two kinds: a direct subtype of `AbstractAlgorithm` stands alone, and a
      `DefaultAlgorithmExtension` inherits the default rules and dependencies for whatever it
      does not define. The inheritance is a second lookup in the untyped `find_*` and
      `dependencies_spec` fallbacks, not subtype dispatch, which would make an override with
      broader inputs ambiguous. An inherited rule runs with `DefaultAlgorithm()` in its `algo`
      slot (`rule_algorithm`). Diagnostics, `check_rules` and `list_rules` see inherited rules.
      The fallback call is inferred, JET-clean and 0 bytes on 1.10 and 1.13
      (`gate:routing-macros`). Tests: `algorithm:*`; the tests that used `VMP` as a second
      algorithm now name what they need (`MixtureVMP`, `Standalone`, `Alternative`,
      `FixedPartition`, `ToyDelta`)
- [x] the lib packages and scripts follow: the standard nodes drop `algorithm = BP`;
      `NormalMixture` runs under its own exported, standalone `NormalMixtureVMP`, documented as
      always variational whatever the factorisation; TestUtils' test rules drop
      `algorithm = VMP`, since the verification reference was already chosen from the input
      kinds, not the algorithm; `check.jl` calls `DefaultAlgorithm()`. Labels naming v6's inference
      modes (`check.jl`, the fixture ids) describe the mathematics and stay. Every suite and
      every v6 comparison is unchanged
- [x] the documents follow: PLAN § Dispatch axes states the model, and its examples, queries,
      dependencies, purity, #3, #4 and Delta passages match it. DISCUSSION has §3.20 and
      Corrections 23, with notes at the older passages. This file's conventions, contracts
      and #3/#4 notes are updated too, as are lib/README, the INVENTORY notes for `Require*` and
      the CHANGELOG. Mathematical uses of BP and VMP stay

### Step 3 — the slice's rules, ported

The porting list is exact, not estimated: `compat/v6-comparison/slice_rule_inventory.jl`
records every message rule, marginal rule and average energy the seven slice models *(recorded in step 0)* select
in v6, with the input types the selected rule declares. Deduplicated, it is:

| node | message rules | marginal rules | average energies |
|---|---|---|---|
| `NormalMeanVariance` | `:out` ×3, `:μ` ×2 (+ their message-input siblings) | `(:out, :μ)` | singles, `(:out, :μ)` joint |
| `NormalMeanPrecision` | `:out` ×2, `:μ` ×2, `:τ` ×2 | `(:out, :μ)` | singles, `(:out, :μ)` joint |
| `GammaShapeRate` | `:out` | — | singles |
| `Categorical` | `:out`, `:p` | — | singles |
| `Dirichlet` | `:out` | — | singles |
| `NormalMixture` | `(:m, k)`, `(:p, k)`, `:switch` | — | its own |
| `DeltaFn` + `Unscented` | `:out`, `(:in, k)` | `(:in,)` | — (engine) |

Porting conventions:
- a distribution node runs under `DefaultAlgorithm` and its rules omit `algorithm`; they
  combine `m[…]` and `q[…]` as v6's did, and the factorisation decides which apply, through
  the default dependency scheme *(this read `BP` until the algorithm reconciliation above)*;
- a rule taking a non-point-mass `q_v` is ported with ReactiveMP.jl#669 **corrected**, and the
  comparison declares it a `:correction`;
- log scales are kept exactly as v6 has them, gaps included;
- each node gets v6's own tables, verification against its log-density where the tool
  supports the inputs (messages only, or marginals only), and a v6 comparison in
  `compat/v6-comparison/compare_standard.jl`, run locally.

**Renamed on 2026-09-23 (user): the rule keyword `towards` is now `target`**, in the macros,
the interactive functions, the test tooling and every document. See `DISCUSSION.md` §3.19.

Findings so far:
- **Aqua's piracy check does see rule packages**, contrary to `PLAN.md` § Testing's
  "vacuous for rules". Message rules escape it, because their target type carries a
  `Symbol`. But `nodespec`, `nodefunction`, average energies and marginal rules for another
  package's distribution are flagged. `StandardMessagePassingRules` declares the node types
  it defines as owned (`NODES`, passed to `treat_as_own`); that is the design, a rule package
  declaring nodes for distributions it does not own.
- The verification tool takes messages or marginals, not both, so rules mixing the two are
  checked by their tables and the v6 comparison only.

- [x] `NormalMeanVariance`: `out.jl`, `mean.jl`, the `(:out, :μ)` marginal and both average
      energies. 40 cases against v6: agreement everywhere except the seven declared #669
      corrections, and the corrected rules verify against the node definition
- [x] `NormalMeanPrecision`: `out.jl`, `mean.jl`, `precision.jl`, the `(:out, :μ)` marginal
      and both average energies. It needs no correction: `E[τ]` is right for a precision. v6's
      scalar `cholinv` becomes `inv`, and the v6 comparison agrees on every case
- [x] `GammaShapeRate`, `Categorical`, `Dirichlet`: every v6 rule and average energy of the
      three nodes that the slice models select *(not every one, as this line first said: GSR's
      `a.jl`, `b.jl` and marginal, Categorical's two marginals and catch-all `p.jl` rule, and
      Dirichlet's marginal are Phase 5 step 1)*, and v6's private `isonehot` helper. One easy win beyond v6: the softened
      `Categorical(:out)` rule for `q[:p]::Dirichlet` clamped to `[tiny, Inf]`, which turned
      every input into `Float64`, so v6 tested no Categorical rule for type promotion.
      `max(ρ, tiny)` gives the same numbers and keeps the input's precision. The v6
      comparison agrees on all of it, 100 checks in total. Categorical tables check promotion
      in Float32 and Float64 only, because converting a Categorical whose probabilities sum
      to one only approximately into BigFloat fails Distributions' own check
- [x] `NormalMixture`, the design's canary: the node is now this package's own
      `struct NormalMixture end`, with no `{N}`, since the components are the groups' length.
      It is declared `interfaces = [:out, :switch, :m..., :p...]` under its own standalone
      algorithm `NormalMixtureVMP` (`VMP` until the algorithm reconciliation), always
      variational whatever the factorisation, with its
      dependencies, `(:m, k) => (q[:out], q[:switch], q[:p][k])` and so on. Its rules towards
      `(:m, k)`, `(:p, k)`, `:switch` and `:out` and its average energy were univariate here;
      the multivariate branches came in step 5. The switch rule and the
      energy share `normal_mean_precision_energy` with NMP. The v6 comparison agrees on
      every case, with v6 given the aligned member alone where v7 passes the whole group.
      The package's `quality:rules` now asserts that `check_rules` and
      `check_rule_ambiguities` find nothing. That assertion found a base-package bug:
      disjoint rules were reported as ambiguous, which `a0682b5e` fixes
- [x] `Unscented` into `MessagePassingRulesApproximations`: `Unscented` and its aliases,
      `sigma_points_weights`, `unscented_statistics`, `approximate` over means and
      covariances, `smoothRTS` and the shared split/join helpers. **Pure numerics** (user):
      `PLAN.md` said the survivors need only ForwardDiff, Random, LinearAlgebra and
      Distributions, but v6's `unscented.jl` also used ExponentialFamily, for `JointNormal`
      and `NormalMeanVariance`, and FastCholesky. The multi-input joint is now concatenated
      means and a block-diagonal covariance, which is all `JointNormal` contributed. FastCholesky
      stays, as the small numeric dependency #13 expects. The dependencies are LinearAlgebra
      and FastCholesky alone; ForwardDiff comes back with `Linearization` in Phase 6. The
      distribution-level `approximate(::Unscented, f, ::NTuple{N, NormalDistributionsFamily})`
      and `is_delta_node_compatible` move to the Delta rules, and the Delta node stays in
      ReactiveMP for the engine to port in case (d) *(done, in `lib/DeltaMessagePassingRules`)*. v6's numeric tests are ported, and
      `compare_approximations.jl` agrees with v6 to 1e-12 on 22 checks, the mixed
      scalar, vector and scalar joint included

### Step 4 — the engine core, on case (a): the plan

**Goal:** case (a), belief propagation over `NormalMeanVariance` (`bp_iid`, `bp_iid_missing`
and `bp_chain`), running end to end through the new rule system and agreeing with the v6
fixtures. It is a **clean cut** (user, §3.22): no dual path and no transition stage.

1. [x] **Toolchain, Julia 1.13 only.** *Done:* every lib package tests with plain `Pkg.test`,
   `[sources]` wiring the siblings (test-only ones through `[extras]`), `julia = "1.11"`;
   Standard's `test/Project.toml` is gone. `compat/v6-comparison` is resolved on 1.13: v6.5.0
   and RxInfer 5.5.2 run there, the fixtures recorded on 1.10.12 reproduce exactly, and every
   comparison is unchanged. ReactiveMP's own `[deps]` entry lands with item 4, where it is
   first used.
   - ReactiveMP gets `[deps]` + `[sources]` for `MessagePassingRulesBase`, and `[extras]` +
     `[sources]` for `StandardMessagePassingRules` and `MessagePassingRulesTestUtils`, which
     are test-only.
   - Every lib package's `test/Project.toml` or develop step is replaced by `[extras]` +
     `[sources]`, and the Makefile targets become plain `Pkg.test`.
   - `julia = "1.11"`, the `[sources]` minimum, tested on 1.13 only.
   - `compat/v6-comparison` is regenerated on 1.13 and the fixtures re-checked. If v6 will not
     run there, that environment alone stays on 1.10.
2. [x] **Move, don't delete** (user). The v6 rule system and every unported node go to
   `legacy/v6/`, mirroring their old paths, kept for reference and never loaded:
   - code: `src/rule.jl` (`@rule`, `@marginalrule`, `@call_rule`, `@test_rules`), `src/rules/`,
     `src/nodes/predefined/`, `@node`, `src/approximations/` and `ext/`;
   - tests: `test/rules/`, `test/nodes/predefined/` and `test/approximations/`.

   `legacy/` is excluded from the root test filter, since TestItemRunner scans the whole
   package directory, and from the formatter. A `legacy/README.md` says what it is, and that
   Phase 5 empties it as each node is ported. `docs/` is left out of the build until it is
   rewritten. The inventory gate enumerates v6.5.0 from `compat/v6-comparison`, so INVENTORY
   stays the record of where everything goes.

   *Done.* Also moved: the v6 algebra helpers, `src/fixes.jl`'s `ForwardDiff` hot-fix (only the
   approximations needed it), and the tests of all of these. `@node`, its traits, the
   `Require*` dependencies and `@average_energy` left with the pre-step-4 copies of
   `nodes.jl`, `dependencies.jl`, `clusters.jl` and `score/`, kept beside them. `@logscale` is
   gone too: it expanded to v6's rule-scoped `getannotations()`, and a rule now writes
   `annotate!(ann, :logscale, value)`. ReactiveMP's `[deps]` shrank to BayesBase,
   Distributions, LinearAlgebra, MacroTools, MessagePassingRulesBase, Rocket, TinyHugeNumbers,
   TupleTools and UUIDs; the version is `7.0.0-DEV`. The root test filter also skips `lib/` and
   `compat/`, which have suites of their own, and `make docs` refuses with a pointer here.
   Regenerating INVENTORY from v6.5.0 reproduces it byte for byte.
3. [x] **Nodes from `NodeSpec`.**
   - `factornode` takes `(name, index)` interfaces, and clusters as tuples of interface
     names.
   - Arity, aliases and `sdtype` come from the NodeSpec, and ReactiveMP's
     `Stochastic`/`Deterministic` become the base's.
   - `FactorNodeLocalMarginal` carries its member tuple, never a joined name.
   - Activation is generic: `dependencies_spec`, or the default scheme when that is
     `nothing`. `algorithm` replaces `meta` in the activation options.

   *Done, with two parts left to case (c), which did them.* `factornode(fform, interfaces, factorisation =
   nothing)` takes `(name, variable)` and `((group, k), variable)`, names or aliases in any
   order, and puts them in declaration order; the factorisation is tuples of the same keys,
   `nothing` meaning one cluster, and a deterministic node always gets one. Errors name the
   node and the offending key. A local marginal is keyed `:μ` for one interface and `(:out,
   :μ)` for a joint. `FactorNodeActivationOptions(; algorithm, postprocessor, annotations,
   callbacks)` replaced the six positional fields; `rulefallback`, `metadata` and the
   dependency policy are gone. **`activate!` refuses a node with a declared
   `dependencies_spec` or an interface group**, with an error saying so, rather than wiring
   them wrongly: the default scheme is the only one wired, and declared dependencies and
   groups are case (c)'s work. *(Case (c) lifted both refusals.)*
4. [x] **The rule-call path.**
   - `MessageMapping`, `MarginalMapping` and the node score call `find_*` and `execute_rule`
     with `rule_algorithm`.
   - `RuleArgs` come from `getdata`. `Val{:out}` becomes `Target{:out}`, `(Val{:m}, k)` becomes
     `IndexedTarget`, and a cluster becomes a `ClusterTarget`.
   - Rules get `RuleContext(node = …)`, and `RuleAnnotations(m, q, out = AnnotationDict)`
     with the base `annotate!`/`getannotation` defined for `AnnotationDict`.
   - The missing-input short-circuit (`message.jl:684-691`), the callbacks and the pre/post
     annotation processors are kept exactly.

   *Done* (`src/rule_arguments.jl`, `MessageMapping`, `MarginalMapping`, `score/node.jl`). A
   rule's `ann.m` and `ann.q` carry the inputs' `AnnotationDict`s, keyed like the arguments. No
   rule found raises `MessagePassingRulesBase.RuleNotFoundError`, with its near misses; there
   is no fallback. A node's average energy is found with `find_average_energy`; v6's
   decomposition of `NamedTuple` joints went with the generic `score(AverageEnergy(), …)`,
   since `FactorizedCluster` replaces it (deferred to Phase 5 by case (b)).
5. [x] **Free energy in the engine:** `bethe_free_energy(T, nodes, variables)`, porting RxInfer's
   assembly (`reactivemp_free_energy.jl:52-128`). It combines the node scores over the declared
   partition or the factorisation, the variable entropies, and the data/constant degree
   correction, and emits one value per iteration.

   *Done* (`src/score/bethe.jl`): `bethe_free_energy(T, factornodes, variables; algorithm =
   node -> nothing)`, the algorithm per node being the one it was activated with. Node scores
   run over the factorisation; a declared partition arrives with declared dependencies in
   case (c).
6. [x] **A test harness** (`test/engine/`). It builds a graph in RxInfer's order: variables;
   factor nodes in statement order, with interfaces `out, μ, v`; activation; subscriptions;
   and data fed each iteration. It traces with `after_message_rule_call`, produces an
   `EngineTrajectory`, and runs `compare_engine_trajectory` against `bp_iid`,
   `bp_iid_missing` and `bp_chain`.

   *Done* (`test/engine/harness.jl`, `fixtures_tests.jl`). **All three agree with v6 at
   `atol = 1e-9`**: every rule call in order with its result, the log scales of `bp_iid`, the
   posteriors and the free energy (8.288494360822888 and 9.679161779955816 per iteration). As
   RxInfer does, the harness activates data variables with predictions on, subscribes to the
   predictions only of data with a `missing`, and factorises random interfaces jointly and
   each data or constant interface alone.
7. [x] **The engine's own tests** are rewritten on `@define_factor_node` toy nodes:
   `message_tests`, `callbacks`, `variables`, `dependencies`, `clusters` and `nodes`. The
   missing-input pin, the retained-value test and the `Message` benchmark are added here.

   *Done.* `nodes_tests` tests `factornode` itself now; `@node`'s own tests went to
   `legacy/v6/test/nodes/` (deleted in Phase 5 step 9), as the base package tests
   `@define_factor_node`. The v6 tests of
   `Require*` and meta-driven dependencies went with them. New: `MessageMapping` resolving and
   running a rule, with the algorithm and node it receives; `RuleNotFoundError`; the
   **missing-input pin** (no rule call, the pre-rule processors run, the post-rule ones do
   not); the default scheme per factorisation; the case-(c) refusals; and
   `engine:retained-values`, which holds materialised messages and marginals across further
   updates. The root suite was 115 items at step 4, 14 857 tests, all passing, Aqua included.

   **The `Message` benchmark** (`scripts/benchmark_message_representation.jl`, Julia 1.13,
   M-series Mac; the data-feeding loop only, minimum of seven runs, `Message` and `Marginal`
   changed together):

   | graph, 10 iterations | `mutable`, `const` fields | immutable |
   |---|---|---|
   | iid, n = 1000 (equality chain of degree 1001) | 2.9 ms, 5.65 MiB | 3.2 ms, 9.62 MiB |
   | iid, n = 10 000 | 41–45 ms, 56.5 MiB | 49–50 ms, 96.1 MiB |
   | chain, n = 300 | 12.6 ms, 12.5 MiB | 11.7–11.9 ms, 15.7 MiB |

   **`mutable` stays**: about 10% faster and 40% lighter through the equality chain, which is
   where the brief said to measure; immutable wins 6–8% on the chain but allocates 25% more.
   Typed annotations (`Message{D, A}`, brief item 3) are not done: the `AnnotationDict` stays,
   and the retained-value test pins that nothing mutates one after materialisation.

### Cases (b)–(d): what each did

Written after step 4 as what each case still needed, and completed as each was done. Each case
was its own step: one commit, test first, the fixture comparison as its gate, `PHASES.md` and
`CHANGELOG.md` in the same commit.

**Common to all three.** `test/engine/harness.jl` takes a `factorisation` per node
(`node!(…; factorisation)`, default Bethe, with `meanfield_factorisation`) and
`initial_marginals`, each set right after its variable is activated and before any node is, as
RxInfer does (`reactivemp_inference.jl:316-335`); both were added in case (b). Build a graph in
GraphPPL's order, which is RxInfer's activation order: variables as the statements create them,
a statement's constants after its random variable, and data where it is first used
(`vmp_structured` interleaves `x[i]`, `y[i]` and the `0.5` of each `y` node). Posteriors are
subscribed in the order RxInfer's `returnvars` `Dict` iterates (`batch.jl:335-342`): `μ, τ, x`
on 1.13. Each fixture's model, constraints and initialization are in
`compat/v6-comparison/record_engine_fixtures.jl`; `slice_rule_inventory.jl` lists every v6 rule
each model selects, with its input types. Compare with `atol = 1e-9` as for case (a), and
declare any disagreement rather than loosen the tolerance.

**Case (b), VMP — done** (`vmp_meanfield`, `vmp_structured`; NMP, GammaShapeRate and NMV).
Both agree with v6 at `atol = 1e-9`, call by call: 62 and 122 rule calls in v6's order, the
posteriors, and the free energy over five iterations, `vmp_structured`'s non-monotone one
included (`engine:fixture:vmp_meanfield`, `engine:fixture:vmp_structured`). **No engine code
changed**; the harness gained a factorisation per node and initial marginals. A negative
control, a wrong initial `q(τ)`, fails on the free energy, the posteriors and the trace.
- Mean-field needed nothing new: every cluster is a single interface, so rules read marginals
  only. `q(τ)` is initialised.
- Structured, `x[i] ~ NMP(μ, τ)` under `q(x, μ)q(τ)`, was the first joint in a VMP run: the
  `ClusterTarget((:out, :μ))` marginal rule, the `τ` rule reading `q[:out, :μ]`, the joint
  average energy and the joint's entropy in the node score all ran end to end for the first
  time.
- **`FactorizedCluster` (brief item 5) is deferred to Phase 5.** *(Done in Phase 5 steps 1–2;
  the NamedTuple entropy method is deleted, and the files cited below are only in v6.5.0.)* `slice_rule_inventory.jl` shows
  the only marginal rules the slice selects are NMV's and NMP's `(:out_μ)` over two Normal
  messages (`bp_chain`, `vmp_structured`) and Delta's `(:ins)`; each returns one joint, never
  v6's NamedTuple split, which needs a `PointMass` message. So no slice model produces a
  `FactorizedCluster`. The first port that returns one (the `PointMass` variants of those
  marginal rules, `legacy/v6/src/rules/normal_mean_{variance,precision}/marginals.jl`) brings
  the engine work with it: hand each block to a consumer reading that member, and give
  `score(DifferentialEntropy(), ::Marginal{<:NamedTuple})` (v6's split, still in
  `src/score/score.jl`) its `FactorizedCluster` counterpart.
- A Bethe-factorised x-node in the same model stalls rather than failing: loopy BP with no
  initial messages never fires, as in v6, so its posteriors never emit.

**Case (c), the mixture — done** (`normal_mixture`; NormalMixture, Categorical, Dirichlet, NMP,
GammaShapeRate). `engine:fixture:normal_mixture` agrees with v6 at `atol = 1e-9` in the free
energy over five iterations and every posterior, and makes the same 285 rule calls, each in
its iteration with the same result; their order within an iteration differs by decision, below.
- **Declared dependencies are wired** (`declared_dependencies`, `src/nodes/dependencies.jl`): per
  target, each declared input becomes an inbound message, a variable's marginal or, for a tuple
  key, a local cluster's marginal, with the selectors `all`, `aligned`, `all-but-self` and
  custom ones resolved against the target's own index. The default scheme handles groups too.
- **Groups reach a rule as one tuple**, full length, `nothing` for the members not selected.
  An input is labelled by name, cluster tuple or `GroupMember`, and `input_names` folds a
  group's consecutive members into a type-level `GroupInputs{name, n, members}`, which
  `rule_messages`/`rule_marginals` expand in generated code. The node score folds its cluster
  marginals the same way. `ManyOf` and `proxy_type`, v6's group container and signature
  helper, are deleted.
- **A declared free-energy partition** must be the factorisation, block for block (a group's
  name standing for all of its members), or `activate!` raises an error naming the algorithm
  (#9); `bethe_free_energy` then scores over it as it does over any factorisation.
- `activate!` still refused a **joint cluster over group members**; Delta's `(:in,)` was case
  (d), which lifted it for whole groups.
- **#6, the mixture's `reverse(...)`, is resolved by not reproducing it** (user; `DISCUSSION.md`
  §3.24). Inputs are subscribed in declaration order, and in VMP that order is the update
  schedule. v6 subscribed `(out, p₂, p₁, m₂, m₁)` for `:switch`. The group order, precisions
  before means, is the schedule: means first reaches the same optimum in about 13 iterations
  instead of about 7. So `NormalMixture` declares its precisions first, with that reason in its
  docstring. The member reversal changes no value, since members do not depend on each other,
  so the trace is compared per iteration with its order free
  (`compare_engine_trajectory(…; trace_order = :within_iteration)`, new in TestUtils).
- **#7, group indices through integration**: the caller names a member as `((:m, k), variable)`,
  `rule_target` makes it `IndexedTarget{:m}(k)`, and the selectors resolve `k` against it; the
  fixture, where `m[2]`'s messages differ from `m[1]`'s, agrees. No index is derived from
  position. RxInfer passing GraphPPL's `EdgeLabel.index` as `k` is Phase 7.
- The mixture runs under `NormalMixtureVMP()`, which the harness passes per node, as RxInfer's
  per-node algorithm option will.

**Case (d), Delta — done** (`delta_unscented`, and `delta_unscented_static`, recorded for it).
`engine:fixture:delta_unscented` and `engine:fixture:delta_unscented_static` agree with v6 at
`atol = 1e-9`, call by call in order, in the posteriors and in the free energy. Decisions in
`DISCUSSION.md` §3.25.
- **A package of its own** (user): `lib/DeltaMessagePassingRules`, `INVENTORY.md`'s `node:Delta`,
  created early. The node is `DeltaFn{F}`, declared `Deterministic`, `[:out, :in...]`,
  `static_inputs = :fold`. Its algorithm is `DeltaApproximation(; method, inverse)`, v6's
  `DeltaMeta`, with the compatibility guard `is_delta_node_compatible`. The unknown- and
  known-inverse layouts are two `@define_dependencies` on its two forms, with the partition
  `[(:out,), (:in,)]`. The Unscented rules are ported with v6's own tables, and leave
  `legacy/v6/` in this commit. A rule-by-rule v6 comparison is not written: the tables and the
  two fixtures cover the rules; it comes with `Linearization` in Phase 6.
- **The engine owns the function.** `factornode(…; nodefn = f)`; a `static_inputs = :fold` node
  requires it. The group members connected to a constant or to data get no interface, as in
  v6, so they add no point entropy; the others are renumbered `1:n`. The node keeps a
  `StaticFold`, which calls `f` with the latest static values in their places, and implements
  `getnodefn(node, Target(:out))`. Every update of the node, messages and the joint, waits for
  the static inputs (`with_statics`, as v6). The known inverse is the algorithm's, read from
  `algo`, not `getnodefn`'s; the base docstring says so.
- **Deterministic nodes** have two clusters, `out` and the joint over the inputs, whatever the
  factorisation. The joint over a whole group is keyed by its name, `(:in,)`, even for one
  member, and computed by the marginal rule from the messages on every interface (v6's `q_ins`).
  The free energy is minus its entropy; the default scheme for a deterministic node is belief
  propagation over the other interfaces' messages.
- **An empty group selection**, `m[:in][!k]` with one input, reaches the rule as `(nothing,)`
  (`EmptyGroup`).
- `activate!` now refuses only a joint holding **some** members of a group with other
  interfaces.
- TestUtils compares a non-distribution struct output, such as `JointNormal`, field by field.

### Design brief — 2026-09-23

Evidence gathered from the v6 engine, the base package and RxInfer 5.5.2; the reasoning is
in `DISCUSSION.md` §3.19. The whole brief is **signed off** (2026-09-23). The numbered items
were proposals and were accepted as written, except for four that changed when they were
built:
- item 3 became a benchmark;
- item 5 became `FactorizedCluster`;
- item 8's structured model is the one the fixture actually records;
- item 10 was re-sequenced by the clean cut (§3.22).

**Decided**

- **RxInfer adapts, as its own major release — but the adaptation is expected to be
  small.** The engine's machinery is kept, so most of what RxInfer 5.5.2 uses keeps working
  (streams, `set_initial_*`, `new_observation!`, `score`, callbacks and form constraints). What
  breaks is node and rule creation and definition: `factornode(fform, interfaces,
  factorization)` with positional edges, `FactorNodeActivationOptions`, the `@node` traits
  that GraphPPL reads, and the force-marginal plugin's reach into node internals. RxInfer is
  adjusted there. The slice's tests build graphs through the engine's own API, and RxInfer
  5.5.2 records the v6 fixtures.
- **Rocket stays the substrate.** The engine keeps its Rocket observables and is not replaced
  by an explicit scheduler — `PLAN.md` § Package split already lists Rocket as the engine's
  dependency, and v6's emission order stays comparable.
- **Deferred messages materialise exactly as in v6.** A deferred message holds its source
  observables and reads their latest values when it is materialised, then caches
  (`message.jl:447-494`). This is **load-bearing for the correctness of reactive message
  passing** (user), not an implementation detail to improve on: the new engine reproduces it,
  and the v6 fixtures check it. The retained-value criterion therefore starts at
  materialisation — once materialised, a message's value and annotations never change
  underneath a holder.
- **The slice's rules are ported into the real packages**, not throwaway fixtures: NMV, NMP,
  Gamma(ShapeRate), Categorical, Dirichlet and NormalMixture into
  `lib/StandardMessagePassingRules`, each with TestUtils tables and a v6 comparison;
  `Unscented` moves into `lib/MessagePassingRulesApproximations` as pure numerics (see step 3);
  the Delta node and its rules move to `legacy/v6/` in step 4 and are ported for case (d).
  About 45 rules were estimated; **54** were ported: 43 message rules, 2 marginal rules and 9
  average energies.
- **The base package gains a cluster over a whole group** — a joint target and input over
  `:in...`, validated like any other cluster. `validate_dependencies` used to reject it, so
  Delta's `q_ins` joint (`rules/delta/unscented/marginals.jl:4`, `in.jl:3`) could not be
  written; done in step 2, as `q[(:in,)]`. Delta is the first customer.

**Signed off**

1. **Edge identity is explicit.** The engine's construction API takes interfaces as
   `(name, index)` — the index supplied by the caller (GraphPPL's `EdgeLabel.index`), never
   re-derived from neighbour position — and clusters name interfaces, not positions in a flat
   list. That removes #7 and the edge-order trap (`clusters.jl:77-87`, `dependencies.jl:311-320`,
   `mixture.jl:84-95`, `delta.jl:170-179`) by construction.
2. **No per-node `activate!`.** Activation is generic over the `NodeSpec`/`DependenciesSpec`:
   the mixtures' own `factornode`/`activate!`/`collect_latest_*` (`normal_mixture.jl:56-225`)
   become groups plus declared dependencies. Delta's three non-dependency behaviours are
   engine features keyed off the spec: static gating (`static_inputs = :fold`), `q_out`
   aliasing (a singleton cluster's marginal *is* the variable's), and the empty-group case.
3. **`Message` carries typed annotations** (`Message{D, A}`); v6's mutable `AnnotationDict`
   (`annotations.jl:12`), which tests mutate in place, goes. **Whether the struct itself is
   `mutable` with `const` fields or immutable is decided by a benchmark, not by taste**
   (user). v6's `mutable struct … const` fields (`Message`, `message.jl:77`) are deliberate:
   a mutable struct is passed by reference, which can avoid copying. Measure both
   representations through the equality chain and a full slice sweep, and keep whichever
   wins.
4. **`getnodefn(node, target)` is declared in the base package with no methods** and
   implemented by the engine's Delta node, which owns the function and its static fold.
   *(Declared in step 2; the engine's Delta node implements it in case (d).)*
5. **A marginal rule may return a `FactorizedJoint`** (BayesBase) for a cluster that splits,
   replacing v6's NamedTuple returns (`normal_mean_variance/marginals.jl:24`,
   `normal_mean_precision/marginals.jl:58`); the engine distributes it to the cluster members.
   *(Refined while building it, user: a bare `FactorizedJoint` is positional and cannot say
   which members a block covers, so v6's partial splits like `(out_μ = …, v = …)` would be
   lost. The result is a `FactorizedCluster`, the member-tuple labels over a `FactorizedJoint`,
   which remains the distribution. See step 2.)*
6. **The engine assembles Bethe free energy itself** — per node, average energy over the
   declared partition minus entropies, and per variable, the degree-weighted entropy — as a
   stream per iteration. In v6 the assembly is RxInfer's (`reactivemp_free_energy.jl:52-128`);
   the engine needs it to run the slice, and RxInfer later consumes it.
7. **First-cut scope.** In: annotations (log scale), rule-call callbacks (enough to record
   emission order), missing inputs, `RuleNotFound` as an error naming the near misses. Out,
   to Phase 7: form constraints, stream postprocessors and schedulers,
   `InputArgumentsAnnotations`, user fallbacks, buffer reuse (#10: the first cut always
   allocates fresh results).
8. **Slice models.** (a) BP: `x ~ NMV(c, c)`, `y ~ NMV(x, c)`, `y` observed. (b) VMP:
   `μ ~ NMP`, `τ ~ GammaShapeRate`, `y[i] ~ NMP(μ, τ)`, mean-field. The structured case, as the
   `vmp_structured` fixture records it, is a different model: `x[i] ~ NMP(μ, τ)`,
   `y[i] ~ NMV(x[i], 0.5)`, with `q(x, μ)q(τ)`. (c) `y[i] ~ NormalMixture(z[i], (m₁, m₂), (p₁, p₂))`, `z[i] ~ Categorical(π)`,
   `π ~ Dirichlet`, NMP/GammaShapeRate priors. (d) `x ~ NMV`, `z := f(x)` with Unscented,
   `y ~ NMV(z, c)` observed. The stretch case, `f(x, c)` with a static input, had no fixture;
   case (d) recorded one, `delta_unscented_static`, `z := f(2.0, x, s)`.
9. **Fixtures (Step 0).** `compat/v6-comparison` adds RxInfer 5.5.2 (it accepts ReactiveMP
   6.5; installed locally). For each slice model: free energy per iteration, posteriors, and
   the trace (`RxInferTraceCallbacks`, `trace.jl:156-161`) of every `AfterMessageRuleCallEvent`
   — edge, result and log scale, in order (`message.jl:730`). A trajectory-shaped fixture type
   joins `MigrationRecord` in TestUtils. Log scales are compared explicitly, with a tolerance.
   *(Done as Step 0; see above.)*
10. **Order of work**, one commit per step, test first:
    - step 0: the fixtures;
    - step 1: this design session;
    - step 2: the base-package additions (group cluster, `getnodefn`, `FactorizedCluster`
      returns);
    - step 3: the rule ports;
    - *(then the algorithm reconciliation, which was not planned)*;
    - step 4: the engine core on case (a), as a **clean cut** (§3.22), with free energy;
    - then (b), (c) and (d), each its own step.

    *(An earlier version of this item kept v6's rule system alive beside the new path until
    Phase 5 deleted it per directory. The clean cut moves it to `legacy/v6/` in step 4
    instead.)*

### Exit criteria
- [x] v6 fixtures recorded for the slice models (Step 0), before any v6 code is deleted —
      `compat/v6-comparison/fixtures/engine/`, re-checked by `record_engine_fixtures.jl --check`
- [x] a working end-to-end inference in the new engine over the slice: ordinary belief
      propagation, structured VMP, a mixture (variadic group), and a delta node — cases (a)–(d)
- [x] free energy agrees with the recorded v6 trajectories on the same models — `bp_iid`,
      `bp_chain`, `vmp_meanfield`, `vmp_structured`, `normal_mixture`, `delta_unscented` and
      `delta_unscented_static`
- [x] annotations and log scales agree with the recorded ones, compared explicitly (see the
      annotation gate in `PLAN.md`), **where v6 records them** (`bp_iid`). Elsewhere v6's
      behaviour is preserved, gaps included — `engine:fixture:bp_iid`
- [x] a retained-value test: hold a materialised message across several updates and confirm
      neither its value nor its annotations change underneath you (deferred messages materialise
      as in v6, so the guarantee starts at materialisation) — `engine:retained-values`
- [x] the missing-input path pinned by a test (rule not called, post-rule processors skipped)
      — `MessageMapping short-circuits a missing input`, and `bp_iid_missing` end to end
- [x] mixture emission order agrees with the recorded v6 order (#6) — *resolved by decision
      instead (user, §3.24): the same calls with the same results in each iteration, in
      declaration order; v6's member reversal is not reproduced and changes no value*
- [x] edge order and group indices preserved through integration (#7) — the caller's `(:m, k)`
      reaches the rule's `k`; `engine:fixture:normal_mixture`
- [x] the v6 rule system and every unported node moved to `legacy/v6/` (step 4); the engine
      keeps only the new rule path; ReactiveMP depends on `MessagePassingRulesBase`
- [x] Delta's own algorithm designed (the method and inverse, like `DeltaMeta`), and
      `getnodefn` implemented by the engine's Delta node (case (d)) — `DeltaApproximation` in
      `lib/DeltaMessagePassingRules`; `getnodefn` on the engine's `FactorNode`
- [x] the `Message` representation chosen by benchmark (mutable with `const` fields vs
      immutable), with the numbers recorded — `mutable` stays; numbers under step 4, item 7

Do not start Phase 5 until this passes.

---

## Phase 5 — `StandardMessagePassingRules` — **DONE**

**Goal:** standard distribution nodes plus arithmetic (`+`, `-`, `*`, dot).

Phase 4.5 already ported the rules the slice's six nodes (NMV, NMP, GammaShapeRate,
Categorical, Dirichlet, NormalMixture) needed, and moved everything else to `legacy/v6/` (step
4). This phase ports the rest from there, emptying `legacy/v6/` node by node, with v6.5.0 from
`compat/v6-comparison` as the oracle. The rules the 4.5 ports found go into `MIGRATION.md`
as the ports go: `towards` → `target`, dropping `BP`/`VMP`, NamedTuple marginals → `FactorizedCluster`,
`NormalMixture{N}` → groups, promoted pass-through blocks, and #669.

### Entry brief — signed off 2026-09-23

**Scope, counted** (`legacy/v6/src/rules/` and `nodes/predefined/`, destinations from
`INVENTORY.md`). 36 directories go to `standard`: 250 message rules, 82 marginal rules and 38
average energies, of which Phase 4.5 ported 43, 2 and 9. **Left: 207, 80 and 29.** About 50 of
the marginal rules return a NamedTuple, which becomes a `FactorizedCluster`. The other 14
directories are Phase 6's. PLAN's "384 + 106" counted a commented-out rule and hand-written
methods; line-start `@rule`/`@marginalrule` gives 380 + 105 in all.

**Decided (user):**
- **Port by hand or by agent, without a transform tool.** PLAN called for a JuliaSyntax
  transform run with v6 as an oracle. The slice's 54 rules were ported by hand, and the
  remaining work is dominated by what a tool would only flag: `FactorizedCluster` returns,
  algorithm questions, helper extraction and new tests. Each directory is instead gated by:
  - v6's own tables, ported as `@test_message_update_rule`/`@test_marginal_update_rule`/
    `@test_average_energy`;
  - a case per rule in `compat/v6-comparison/compare_standard.jl` through
    `compare_with_reference`, any disagreement declared with its reason;
  - `@verify_message_update_rule` against the node's log-density, where the inputs allow;
  - `check_rules` and `check_rule_ambiguities` finding nothing (`quality:rules`).
  `MIGRATION.md` is written from what the ports find, its pairs as doctests; the requirement
  that tool and guide derive from one source goes with the tool.
- **A rule that called another rule calls a helper function instead.** The shared
  mathematics becomes an ordinary function both rules call, as `NormalMixture` already reuses
  `normal_mean_precision_energy`: subtraction → addition, AND/OR `in2` → `in1`, the
  multiplication and dot-product self-calls, later GCV → NMV. No rule lookup from inside a
  body.

**Order of work**, one commit per step or per node, each with `PHASES.md` and `CHANGELOG.md`:
1. **Finish the slice nodes** — *done*. NMV (`var.jl`, 5 marginals), NMP (5 marginals),
   GammaShapeRate (`a.jl`, `b.jl`, its marginal), Categorical (2 marginals, the catch-all
   `p.jl`), Dirichlet (1 marginal): 8 message rules and 14 marginal rules. Their rule
   directories, node files and tests have left `legacy/v6/`; NormalMixture's stays for its
   multivariate branches (step 5). Findings:
   - every split marginal is a `FactorizedCluster`, its pass-through blocks converted to the
     promoted float type by `promoted_cluster` (`src/helpers.jl`), and passes the tables'
     promotion checks;
   - NMV's point-mass `(:out, :μ)` variants read `q_v` as v6 did, so they carry #669 too, and
     are corrected and declared like the others;
   - `GammaShapeLikelihood` is Standard's own type now (exported), with its product;
   - `DomainSets` joins Standard's dependencies, for the belief-propagation messages towards
     NMV's `v`, which are log-densities on the half line; TestUtils cannot compare those by
     value, so the tables and the v6 comparison evaluate them at points;
   - the comparison reads v6's NamedTuple marginals as clusters by matching the node's own
     interface names (`v6_cluster_blocks`, `V6Oracle.jl`): 163 checks agree, the #669 ones
     declared;
   - the committed comparison manifest pinned the lib packages by absolute paths from one
     machine; they are relative now, as its README says.
2. **Engine: distribute a `FactorizedCluster`** — *done*. A joint input whose value is a
   `FactorizedCluster` reaches the rule as its blocks, a one-member block as that member's
   marginal (`q[:out]`) and a larger one as a joint (`q[:out, :μ]`), each carrying the joint's
   annotations. It is decided in the generated `rule_marginals` from the labels in the
   cluster's type, so it costs nothing at run time, and serves message rules and the average
   energy alike. v6 decomposed its NamedTuple joints for the average energy only; a message
   rule reading one had no v6 path. The entropy needed no engine code: BayesBase's
   `FactorizedJoint` sums its blocks, so v6's `score(DifferentialEntropy(),
   ::Marginal{<:NamedTuple})` is deleted. Tests: `engine:factorized-cluster:arguments`, and
   `engine:factorized-cluster:graph`, where `x := copy(1.0)` sends a point mass into NMV's
   `q(out, μ)`, which splits, and the node's free energy reads the blocks. No v6 fixture can
   check it, since v6 could not run a message rule on a split joint.
3. **Univariate distributions** — *done*. Beta, Bernoulli, Gamma, GammaInverse, HalfNormal,
   Poisson, Uniform and Uninformative: 24 message rules, 6 marginal rules and 10 average
   energies, with v6's own tables and node tests, and `compare_standard.jl` now agrees on 241
   checks with no new disagreement. `HalfNormal` and `Uninformative` are Standard's own
   exported types. The eight nodes have left `legacy/v6/`. *(Corrected after the Phase 5
   review: the Gamma and GammaInverse average energies had been ported with v6's math errors,
   E[x]/E[θ] for E[x/θ] and θ/E[x] for E[θ/x], and v6's node tests pinned the wrong values.
   Both are corrected, verified by hand and by Monte Carlo, and declared in the v6 comparison;
   ReactiveMP.jl#672.)* Two product questions, decided with the user:
   - **`Uninformative`'s product.** v6 made it the identity with generic
     `prod(::PreserveTypeProd{T}, ::Uninformative, ::T)` methods, which were eleven method
     ambiguities with BayesBase. Standard now returns its own strategy, `UninformativeProd`,
     from `default_prod_rule`, so its `prod` methods cannot collide, and settles the remaining
     overlaps with BayesBase's own rules (lazy products, mixtures, terminal arguments) by
     writing their intersections: **0 ambiguities**, and Aqua's check stays on. The weakness is
     that those intersections encode BayesBase's rules, so a new one there can bring an
     ambiguity back. **Recorded fix, upstream:** BayesBase owns `Uninformative` as a product
     identity, one method per built-in strategy, as it already treats `missing`; Standard then
     only declares the node, and its product methods go.
   - **Uniform(0, 1) × Beta** is the Beta, v6's special case, defined for types neither
     package owns: piracy, kept and declared owned in the quality tests as v6 did, and the
     same upstream candidate (ExponentialFamily).
4. **Logic** — *done*. AND, OR, NOT and IMPLY, Standard's own exported types: 11 message rules
   and 3 marginal rules, with v6's tables and a v6 comparison (`compare_standard.jl`: 269
   checks). AND's and OR's `in2` rules called `in1` in v6; each pair now shares a helper. NOT's
   v6 marginal rule `(:in)` is not ported: the engine's joint over a deterministic node's one
   plain input is that variable's marginal, the same value. And the deterministic default
   scheme, written in Phase 4.5 case (d) and never run end to end, now has a v6 fixture:
   `logic_bp`, a tree of the four nodes closed by a Bernoulli factor, agrees call by call,
   posteriors and free energy included (`engine:fixture:logic_bp`).
5. **Multivariate normals**: MvNormalMeanCovariance, MvNormalMeanPrecision (whose `precision.jl`
   uses the correction strategy), MvNormalMeanScalePrecision and its matrix form,
   MvNormalWeightedMeanPrecision; then NormalMixture's multivariate branches. **Done** (the
   step 5 brief, `DISCUSSION.md` §3.28).
6. **Matrix and Wishart**: Wishart, InverseWishart, MatrixNormal, MatrixNormalWishart,
   MvNormalGamma, MvNormalWishart, DirichletCollection. **Done** (the step 6 brief,
   `DISCUSSION.md` §3.29).
7. **Arithmetic**: `+`, `-`, `*`, `dot`. **Done** (the step 7 brief, `DISCUSSION.md`
   §3.31–3.33). The two questions recorded here are settled there: v6's
   `default_meta = ReplaceZeroDiagonalEntries(tiny)` is the rules' default for an unset
   `ctx.matrix_correction`, not an algorithm, and the sampling rules draw from `ctx.rng`.
8. **Mixtures**: GammaMixture, a clone of NormalMixture; then `Mixture`, hand-written: its
   switch rule builds a `randomvar` for a product with log scale (the `product` context
   service instead), and its rules read incoming log scales (`ann.m`). **Done** (the step 8
   brief, `DISCUSSION.md` §3.34–3.35).
9. **Close**: the migration guide complete, `docs/` rewritten and back in the build, `legacy/v6/`
   holding only Phase 6's nodes. **Done** (the step 9 brief, `DISCUSSION.md` §3.36):
   the guide is the docs page `migration-guides/v6-to-v7.md`, not a `MIGRATION.md`.

### Step 5 brief — the multivariate normals

**Scope, counted** from `legacy/v6/`: 48 message rules, 27 marginal rules (23 of them split,
so `FactorizedCluster`s) and 11 average energies, in five nodes, then NormalMixture's
multivariate branches.

| node | interfaces (aliases) | message | marginal (split) | energies |
|---|---|---|---|---|
| MvNormalMeanCovariance | out, μ (mean), Σ (cov) | 14 | 6 (5) | 2 |
| MvNormalMeanPrecision | out, μ (mean), Λ (invcov, precision) | 16 | 8 (7) | 4 |
| MvNormalMeanScalePrecision | out, μ (mean), γ (precision) | 8 | 6 (5) | 2 |
| MvNormalMeanScaleMatrixPrecision | out, μ (mean), γ (scale), G (matrix) | 8 | 6 (5) | 2 |
| MvNormalWeightedMeanPrecision | out, ξ (xi, weightedmean), Λ (invcov, precision) | 2 | 1 (1) | 1 |

**Decided (user): the matrix correction is a context service, `ctx.matrix_correction`**, beside
`linalg` (the future Cholesky strategy) and `rng`. It is not an algorithm.
- `MessagePassingRulesBase` gains the `RuleContext` field and the `CONTEXT_SERVICES` entry
  `matrix_correction`, documented as a strategy from MatrixCorrectionTools
  (`ReplaceZeroDiagonalEntries`, `AddToDiagonalEntries`, `ClampDiagonalEntries`, …).
- A rule declares `ctx = (:matrix_correction,)` and calls `correction!(ctx.matrix_correction,
  M)`. `nothing` is the identity, which is also v6's default for MvNormalMeanPrecision, whose
  `default_meta` was `nothing`.
- The engine passes `nothing` until Phase 7 lets a user set it per node, through
  `FactorNodeActivationOptions` and RxInfer.
- Step 7 settles how `*` and `dot` keep v6's default, `ReplaceZeroDiagonalEntries(tiny)`, when
  the service is `nothing`.
- MatrixCorrectionTools 1.2 joins Standard's dependencies, with LinearAlgebra and
  FastCholesky.

**Defaults for the step**, open to the user's correction:
- **Order:** one commit per node, MvNormalMeanPrecision before NormalMixture's branches, which
  call its energy.
- **Helpers:** `diageye` becomes a Standard helper, and INVENTORY's destination for it changes
  from `base` to `standard`. The 2d×2d joint-precision builder, which v6 writes five times,
  and the ΔΔ' block computation, which it writes eight times, become helpers too.
- **Not ported here:**
  - the two `TerminalProdArgument` marginal rules, which are BIFM's, stay in `legacy/v6/` for
    Phase 6's `node:BIFM`;
  - the belief-propagation rules v6 lacks for the two scale nodes are not invented;
  - v6's `@allocated` "Performance" tests are dropped.
- **v6's mistakes:**
  - MvNormalWeightedMeanPrecision's marginal returns the keys `m_out`, `m_ξ`, `m_Λ` in v6. It is
    ported with the right blocks and declared, with a special case in `as_v7`.
  - Any other disagreement a hand-derived check finds is declared, as #669 and #672 were.
- **Kept from v6:** the `Wishart` specialisations, which read `params(q_Λ)` rather than build
  the mean matrix.
- **Float types:** the energies use one convention, `promote_paramfloattype`; v6 mixed it with
  `promote_samplefloattype`.
- **Inverses:** a scalar's expectation of the inverse is `mean(inv, q)`, a matrix's
  `mean(cholinv, q)`, and a matrix is inverted with FastCholesky's `cholinv`, for a symmetric
  positive-definite result without a general LU (v6's own convention for matrices).
- **MvNormalMeanScaleMatrixPrecision** is not a distribution type in ExponentialFamily 2.6: its
  constructor returns an `MvNormalMeanPrecision`. It serves as the node type only.
- **Tests:** v6 has no tables for any of the 27 marginal rules, so all get hand-derived cases.
  `@verify_message_update_rule` is used where the inputs allow, and its limits with Wishart
  and InverseWishart messages are recorded. TestUtils' default promotion checks (Float32,
  BigFloat) are new for these nodes.
- **Progress:**
  - *MvNormalMeanCovariance — done.* The base package has the `matrix_correction` service, an
    optional one: `nothing` is a setting, so `missing_services` never reports it.
    MvNormalMeanCovariance's 14 message rules, 6 marginal rules and 2 average energies are ported,
    with the helpers `diageye`, `coupled_precision`, `difference_moment` and
    `variational_covariance`. v6's variational rules and `(:out, :μ)` marginals took E[Σ] for
    a `q_Σ`; naive VMP gives E[Σ⁻¹]⁻¹, as v6's own energy for the node does. They are corrected
    and declared (ReactiveMP.jl#673, the multivariate #669). `compare_standard.jl` agrees on 319
    checks, the corrections declared. LinearAlgebra and FastCholesky join Standard's
    dependencies.
  - *MvNormalMeanPrecision — done.* 16 message rules, 6 marginal rules (v6's two for BIFM's
    `TerminalProdArgument` stay in `legacy/v6/` for Phase 6) and 4 average energies, two of
    them one method with a Wishart specialisation. The Λ rules pass their scale matrix through
    `ctx.matrix_correction`, tested with `ReplaceZeroDiagonalEntries`; MatrixCorrectionTools
    joins Standard's dependencies. A `q_Λ` rightly contributes E[Λ], so nothing differs from
    v6: `compare_standard.jl` agrees on all 363 checks.
  - *MvNormalWeightedMeanPrecision — done.* 2 message rules, 1 marginal rule and the average
    energy. v6's marginal keyed its blocks `m_out`, `m_ξ`, `m_Λ`, which broke the node's free
    energy for that cluster; the port labels them `out`, `ξ`, `Λ` (ReactiveMP.jl#674), and the
    comparison reads v6's keys without the prefix. The energy's E[log |Λ|] is `mean(logdet, ·)`,
    which v6's `chollogdet` equals for a point mass. 371 checks agree.
  - *MvNormalMeanScalePrecision — done.* 8 message rules (variational only, as in v6), 6
    marginal rules and 2 average energies; nothing differs from v6 (399 checks).
  - *MvNormalMeanScaleMatrixPrecision — done.* 8 message rules (variational only, as in v6), 6
    marginal rules and 2 average energies. The Gaussian-message rules share a
    `series_precision` helper, and the `G` rule symmetrises its scale so a generic-float
    `cholinv` passes `Wishart`'s check. The comparison's joint cases take the marginals outside
    the joint too; nothing differs from v6 (431 checks).
- **NormalMixture — done, which closes step 5.** Its multivariate path goes through helpers
  dispatching on `variate_form`: `mixture_component_energy` (the switch rule and the energy,
  sharing `mv_normal_mean_precision_energy`), `mixture_precision_likelihood` (a Wishart with
  1 + z + d degrees of freedom), and `promote_variate_type` for the `(:m, k)` and `:out`
  messages. v6's Float64 lock (`init = 0.0`) is gone. v6's multivariate tables are ported
  except two `(:p, k)` cases whose precisions are not positive definite; the `GaussianMixture`
  alias comes along. The comparison adds six multivariate rule cases and the energy; 445
  checks agree. Found on the way: every multivariate energy was a Float64 whatever its inputs
  (`d * log2π`), fixed in its own commit; ExponentialFamily's InverseWishart has the same bug
  upstream (ExponentialFamily.jl#322).

### Step 6 brief — Matrix and Wishart

**Scope, counted** from `legacy/v6/`: 28 message rules, 4 marginal rules (each over the whole
node with point-mass inputs, so `FactorizedCluster`s) and 6 average energies, in seven nodes.
Every node's type is ExponentialFamily's or Distributions', so Standard defines no type. None
uses `meta`, a correction strategy, an RNG, a rule-to-rule call, `ManyOf` or
`TerminalProdArgument`, and none has `@allocated` tests.

| node | interfaces (aliases) | message | marginal | energies |
|---|---|---|---|---|
| Wishart | out, ν (df), S (scale) | 4 | 1 | 1 |
| InverseWishart | out, ν (df), S (scale, Ψ) | 4 | 1 | 1 |
| MatrixNormal | out, M (mean), U (rowcov), V (colcov) | 14 | 1 | 1 |
| MatrixNormalWishart | out, M (mean), U (rowcov), V (scale), ν (dof) | 1 | 0 | 1 |
| MvNormalGamma | out, μ, Λ, α, β | 2 | 0 | 1 |
| MvNormalWishart | out, μ (mean), W (scale), λ, ν | 1 | 0 | 0 |
| DirichletCollection | out, a | 2 | 1 | 1 |

**Decided (user): `public_equivalent`** replaces the engine's `to_marginal` (§3.29). It maps an
efficient working type to the public type users expect for the same distribution
(`WishartFast` → `Wishart`, `InverseWishartFast` → `InverseWishart`), the identity by default.
The engine applies it to every marginal it forms, in `as_marginal`, so downstream rules
receive the result as well as users. `MessagePassingRulesBase` owns it for now, documented with
the working/public split and its uses; Standard adds the two Fast methods in the Wishart
commit, which also removes `to_marginal`. BayesBase owning it, with ExponentialFamily
extending it, is recorded for Phase 8.

**Defaults for the step**, open to the user's correction:
- **v6's mistakes**, corrected and declared, each with an issue tagging @Nimrais:
  - Wishart's `:out` with a `q_S` takes `inv(mean(q_S))`; naive VMP needs E[S⁻¹],
    `mean(cholinv, q_S)`, which v6's own energy uses (`rules/wishart/out.jl:11-12`);
  - MvNormalGamma's `:out` with a `q_μ` drops ½ tr(E[Λ] Cov μ) from the rate
    (`rules/mv_normal_gamma/out.jl:6-8`);
  - any other disagreement a hand-derived check finds.
- **Refused rather than guessed:** where v6 accepts `Any` but its math holds for point masses
  only, the port narrows the signature, so other inputs find no rule instead of an E[f(x)]
  computed as f(E[x]): MatrixNormalWishart's energy (`nodes/predefined/matrix_normal_wishart.jl:29-39`),
  and MatrixNormal's energy, whose `isa MatrixNormal` checks silently drop the second moments
  of any other type (`matrix_normal.jl:34,38`).
- **Kept from v6:**
  - MatrixNormal's BP `:out` and `:M` rules returning a vectorised `MvNormalMeanCovariance`: a
    sum of Kronecker covariances is not a Kronecker product, so no MatrixNormal message exists;
  - the improper `InverseWishartFast` likelihoods towards `U` and `V` (degrees of freedom
    `p - n - 1` and `n - p - 1`), which v6's tables assert;
  - MvNormalWishart's `W` alias `scale`, a model-spec name, although ExponentialFamily's `scale`
    of an MvNormalWishart is κ, the `λ` interface;
  - no rules or energies invented where v6 has none: MatrixNormalWishart and MvNormalWishart
    are point-mass only, and MvNormalWishart has no energy.
- **Helpers:** v6's `mul_trace` is not ported; `dot(A', B)` computes tr(AB) without the
  product. The energies put their constants in the result's float type (`gaussian_energy`
  where it fits); v6's `d * log(2)`, `d * (d - 1) / 2 * logπ`, `0.5 *`, `.- 1.0` and
  `n * p * log2π` all make Float64. The float-type test of step 5 extends to these nodes; its
  InverseWishart `@test_broken` stays until ExponentialFamily.jl#322.
- **Inverses:** `cholinv`, as in step 5; v6's `inv` in Wishart's `:out` goes.
- **Tests:** v6's tables, except invalid inputs (MvNormalWishart's ν = 1 at d = 2), dropped
  with a note. Hand-derived cases for the four marginal rules and for DirichletCollection,
  which has no rule tests, rank 2 included, which is DiscreteTransition's. MatrixNormalWishart's
  Monte-Carlo energy test (`rtol = 0.1`) becomes a closed-form case, and v6's `to_marginal` TODO
  a `public_equivalent` value test. Comparison cases in `compare_standard.jl` for every node.
- **Order:** one commit per node: Wishart (with `public_equivalent`), InverseWishart,
  DirichletCollection, MvNormalGamma, MvNormalWishart, MatrixNormal, MatrixNormalWishart.
- **Consumers elsewhere**, all Phase 6 and using the types only, so nothing waits on them:
  ConjugateAR (MvNormalGamma), DiscreteTransition (DirichletCollection) and
  ContinuousTransition (WishartFast, and so `public_equivalent`).
- **Progress:**
  - *Wishart — done.* `MessagePassingRulesBase` has `public_equivalent`, the identity by
    default; the engine's `as_marginal` calls it where it called `to_marginal`, which is gone,
    and Standard adds the methods for `WishartFast` and `InverseWishartFast` (Aqua's piracy
    check treats the two as owned). 4 message rules, 1 marginal rule and the energy. v6's
    variational `:out` rules took E[S]⁻¹ for a `q_S`; they use E[S⁻¹], corrected and declared
    (ReactiveMP.jl#675). The energy's constants are in ν's float type. 461 checks agree, the
    correction declared.
  - *InverseWishart — done.* 4 message rules, 1 marginal rule and the energy; the marginal rule
    takes any InverseWishart message, not only v6's `InverseWishartFast`. Nothing differs from
    v6 (475 checks). ExponentialFamily's E[out⁻¹] fails for a BigFloat InverseWishart, beside
    its Float64 E[log |out|]; both are on ExponentialFamily.jl#322, so the energy table runs
    in Float64 only.
  - *DirichletCollection — done.* 2 message rules, 1 marginal rule and the energy. v6 had no
    rule tests: the cases are hand-derived, and the energy is checked against the Dirichlet
    node's, column by column, for ranks 2 and 3. Nothing differs from v6 (485 checks).
  - *MvNormalGamma — done.* 2 message rules and the energy. v6's variational `:out` dropped
    tr(E[Λ] Cov μ)/2 from the rate for a `q_μ`; corrected and declared (ReactiveMP.jl#676).
    ExponentialFamily's constructor keeps each parameter's float type, so the rules promote
    them to one. The energy is checked against ExponentialFamily's entropy, which it equals
    when the prior is q(out). 495 checks agree, the correction declared.
  - *MvNormalWishart — done.* Its one message rule, for known parameters, promoting them to
    one float type; no energy, as in v6. v6's test used ν = 1 at d = 2, no Wishart; the port's
    cases use valid ones. It agrees with v6 (497 checks).
  - *MatrixNormal — done.* 14 message rules, 1 marginal rule and the energy. The second-moment
    terms go through `row_scatter` and `column_scatter`, dispatching on a point mass or a
    MatrixNormal, where v6 used `isa` checks that dropped any other type's moments; the energy
    and the mean-field rules take `Union{PointMass, MatrixNormal}`, so v6's MatrixNormal-only
    `:out` and `:M` rules now also accept a known `M` or `out`. `mul_trace` is `dot(A', B)`.
    Nothing differs from v6 (531 checks).
  - *MatrixNormalWishart — done, which closes step 6.* Its one message rule, for known
    parameters, promoting ν with the matrices, and the energy. v6's energy took the parameters as
    `Any` and used f(E[x]) for E[f(x)]; they are point masses here, where its formula is exact.
    v6's Monte-Carlo energy test (`rtol = 0.1`) becomes a closed-form one: the energy split
    into its MatrixNormal factor, written out, and its Wishart factor, from the Wishart node's
    own energy. 535 checks agree.

### Step 7 brief — Arithmetic

**Scope, counted** from `legacy/v6/`: 79 message rules, all belief propagation (3 of them only
throw), 25 marginal rules and no average energies, in four Deterministic nodes. The pattern is
the logic nodes' (step 4): belief-propagation rules, a marginal over the inputs, no energy.

| node | interfaces | message | marginal | notes |
|---|---|---|---|---|
| `+` | out, in1, in2 | 30 (out 15, in1 14, in2 1) | 10 (`(:in1, :in2)`) | a generic `convolve` rule; BLAS specialisations |
| `-` | out, in1, in2 | 3 (catch-alls calling `+`) | 10 | |
| `*` | out, A, in | 39 (out 15, in 15, A 9) | 3 (`(:A, :in)`) | `ReplaceZeroDiagonalEntries(tiny)` by default; 2 sampling rules; 11 `@logscale` |
| `dot` | out, in1, in2 | 7 (3 `error`s pointing to SoftDot) | 2 | `ReplaceZeroDiagonalEntries(tiny)` by default |

**Decided (user):**
- **The functions are the nodes** (§3.33): `@define_factor_node(node = +, …)`, dispatching on
  `typeof(+)`. Aqua's piracy check lists the four function types as owned. If the `+` port
  finds a function node needs a hack, it stops and switches to types (`Addition`, …) that a
  base hook `node_type` translates to.
- **An unset `ctx.matrix_correction` is the rule's default** (§3.31): `nothing` means not set,
  `matrix_correction(ctx, default)` in the base package falls back to the rule's default
  (`ReplaceZeroDiagonalEntries(tiny)` for `*` and `dot`, none for MvNormalMeanPrecision, whose
  `precision.jl` switches to the helper), and `NoCorrection()` is an explicit identity.
- **Sampling draws from `ctx.rng`** (§3.32): the rules declare `ctx = (:rng,)`, the engine's
  three `RuleContext`s pass `Random.default_rng()` until Phase 7, and the 3000 draws stay.
  Tests use a `StableRNG`, never a number drawn from the default RNG, whose stream changes
  between Julia versions.
- **The live `*` `:in` log-scale is corrected and declared:** v6's `-logdet(a)` on a scalar
  throws for a < 0 and is off by a factor d; it is −d·log|a|.

**Defaults for the step**, open to the user's correction:
- **v6's mistakes**, corrected and declared, each with an issue tagging @Nimrais:
  - `+` `:in1` for two BLAS `MvNormalWeightedMeanPrecision`s computes μ_in2 − μ_out
    (`rules/addition/in1.jl:88`); `-` `:out` and `:in2` reach it through their redirects.
    Untested in v6.
  - `-`'s marginals shift `in2` by out − in1 where `in2 = in1 − out`
    (`rules/subtraction/marginals.jl:31,103,121,134`); the `-` `:in2` message rule is right,
    and v6's tests assert the wrong value.
  - `*` `:A`'s sampled message weights each sample by |x|, the density of out/in, where the
    message is ∫ p_out(a·x) p_in(x) dx, as v6's own analytic Gaussian `:A` rule computes
    (`rules/multiplication/A.jl:113`). The sampled `:out` rule is right.
  - the `*` `:in` log-scale above.
- **Not ported:**
  - `*` `:in` for `(m_A::Normal, m_out::PointMass)`, which swaps `A` and `out` (`in.jl:156`);
    the message for in = c/A is not Gaussian. Unreachable in v6.
  - the rules that compute in·A for A·in with a matrix operand (`out.jl:43`, `A.jl:15-24`,
    `A.jl:54-67`); refused, recorded in MIGRATION.md and in the `*` issue. Scalar and
    `UniformScaling` operands, which commute, keep their swaps as helpers.
  - `+`'s BLAS specialisations: the generic rules compute the same, and one carried the sign
    error; performance is RxInferBenchmarks' job.
  - the `(Any, Any, meta::Any)` catch-alls of `-` and `+ :in2`, the source of the recorded
    Aqua ambiguities: `-` is written with `+`'s helpers and a negation, `+ :in2` shares
    `:in1`'s helper, and `*`'s commuting swaps and `dot`'s `:in1` → `:in2` are helpers too.
- **Collapsed:** v6 dispatches on argument order, so `*`'s `(m_A, m_out)` rules (`in.jl:112-154`,
  `@logscale 0`, moment outputs) are reachable only through `@call_rule`. Arguments are keyed
  by name here, so each input set has one rule: the live one (`in.jl:40`, weighted-mean output,
  the correction, the corrected log-scale).
- **Kept:** `+`'s generic `convolve` rule; `dot`'s three SoftDot hints, more useful than a
  `RuleNotFound`; `*`'s rank-1 `(PM{Vector}, UniN)` `:out` rule, with v6's TODO restated;
  `dot`'s rank-1 `:in2` precision, through the correction; `+` and `-`'s stacked joint over
  `(:in1, :in2)` as one joint marginal, the point-mass marginals as `FactorizedCluster`s;
  `dot`'s marginal narrowed to the univariate `m_out` its `:in2` rule handles.
- **Numerics:** `0.5 * log(2π …)` becomes `log2π`; `besselmod` becomes float-type generic (no
  `term2 = 0.0`, no Int `factorial(2n)` overflow, no Irrational×Int), with v6's truncation and
  jitter; the sampled closures keep v6's unnormalised sums, a log-scale gap recorded above.
- **Tests:** v6's tables, about 600 cases. The `ContinuousUnivariateLogPdf` messages are
  evaluated at points against quadrature of their defining integrals, not against their own
  closed forms as v6's tautological tests did; the sampling rules the same way with a
  `StableRNG` and a stated tolerance, which is also how the v6 comparison checks them. `*` and
  `dot` with and without a correction, `NoCorrection()` included; the corrected log-scale for
  a < 0 and d > 1.
- **Order:** `+` (with the helpers `-` needs, and the check that function nodes need no hack),
  `-`, `dot` (with `matrix_correction(ctx, default)`), the engine's `rng` default, `*`. Each
  node gets comparison cases, and each commit updates this file and the CHANGELOG.
- **Progress:**
  - *`+` — done.* The function is the node with no hack: `test/engine/function_nodes_tests.jl`
    runs `s = x1 + x2` through the engine, exact posteriors and free energy included, so the
    fallback to types is not needed. Two helpers, `sum_message` and `difference_message`, carry
    v6's per-parametrisation results, with `input_joint` for the joint of two Gaussian inputs;
    `-` will reuse them. The rules take a Gaussian or a point mass, with `convolve` for any
    other two distributions and a rule for two normals resolving the overlap. v6's BLAS
    specialisation for two weighted-mean normals flipped the sign of the `:in1` mean; it is not
    ported, and the difference is declared (ReactiveMP.jl#677). v6's 172 table cases pass,
    with new cases for that specialisation and for `convolve`; Aqua lists `typeof(+)` as owned.
    587 checks agree, the correction declared.
  - *`-` — done.* Its rules are `+`'s helpers in the order the difference needs, where v6 had
    catch-alls calling `+`'s rules, so the Aqua ambiguities they caused are gone. v6's `-`
    marginal took in2's likelihood at out − in1 for a known in1; it is in1 − out, as v6's own
    `:in2` message rule gives, corrected and declared (ReactiveMP.jl#678). v6's 170 table cases
    are ported, 18 of them, which asserted the error, recomputed from the weighted-mean formula.
    The messages with two weighted-mean inputs, which reach #677 in v6 through the redirects,
    are declared too. 639 checks agree, the corrections declared.
  - *`dot` — done.* `MessagePassingRulesBase` has `matrix_correction(ctx, default)`, and its
    context documents `nothing` as not set; MvNormalMeanPrecision's Λ rules read the service
    through it with no default, so they are unchanged. `dot`'s rank-one precision falls back to
    v6's `ReplaceZeroDiagonalEntries(tiny)`, tested with and without a correction, and
    `NoCorrection()` keeping the zero. v6's `meta` in its 102 table cases becomes
    `ctx.matrix_correction`. The SoftDot hints stay, and the marginal takes the univariate
    `out` its message rule handles. Nothing differs from v6 (655 checks).
  - *The engine's `rng` default — done.* Its three `RuleContext`s go through
    `rule_context(node)`, which passes `Random.default_rng()` as `ctx.rng`; Random moves from
    the engine's test extras to its dependencies. A test runs a rule declaring `ctx = (:rng,)`
    through a `MessageMapping`.
  - *`*` — done, which closes step 7.* 26 message rules and 2 marginal rules, down from v6's 39
    and 3: the argument-reversed `:in` rules collapse into the live ones, the wrong
    `(m_A::Normal, m_out::PointMass)` swap is not ported, and the rules that computed in·A for
    A·in with a matrix operand, or `in \ out` for a matrix `in` towards A, are refused.
    Helpers `scaled` and `unscaled` carry the forward and backward messages, the latter through
    the correction (`ReplaceZeroDiagonalEntries(tiny)` by default). Two v6 errors are corrected
    and declared: the sampled messages towards a factor weighted each draw by |y|
    (ReactiveMP.jl#679), and the scalar `:in` log-scale, now −d·log|a| (#680). The sampled
    rules draw from `ctx.rng`; `besselmod` is float-type generic. The closure messages are
    tested against quadrature (HCubature, and StableRNGs for the draws, join Standard's test
    extras), and so is the v6 comparison, with a midpoint rule and a seeded Xoshiro, since the
    oracle environment has neither. 729 checks agree, the corrections declared.

### Step 8 brief — Mixtures

**Scope, counted** from `legacy/v6/`:

| node | message rules | marginal | energies | v6 tests |
|---|---|---|---|---|
| GammaMixture | 4: `:out`, `:switch`, `(:a, k)`, `(:b, k)`, all variational | 0 | 1 | 4 rule items (7 cases); 8 node items, mostly machinery |
| Mixture | 5: `(:inputs, k)` ×2, `:out` ×2, `:switch`, over messages; the two taking `q_switch::PointMass` are dead | 0 | 1, a placeholder | none; 1 node item, machinery |

Both are hand-written in v6 (`GammaMixtureNode{N}`, `MixtureNode{N}`, with their own
`factornode`, `activate!` and functional dependencies); groups replace all of it, as they did
for NormalMixture. The Mixture rules read raw `messages[i]` and incoming log scales, and its
switch rule builds a throwaway `randomvar` for a product's log scale
(`rules/mixture/switch.jl:11`). The engine passes no `product` service yet, never calls
`missing_services`, and does fill `ann.m`; the tables cannot feed incoming annotations.

**Decided (user):**
- **`ctx.product` returns the product's own log scale** (§3.34): the engine's `rule_context`
  passes `product = (l, r) -> (d, compute_logscale(d, l, r))` with `d = prod(GenericProd(), l,
  r)`, and a rule adds the incoming log scales from `ann.m`.
- **Mixture has no average energy** (§3.35): a free energy of a model with one raises
  `RuleNotFound`, naming the node, instead of v6's warning and 0.0.
- **Mixture runs under `MixtureBP`** (§3.35), renamed from the sketches' `MixtureVMP`.
- **A v6 engine fixture, `mixture_bp`**, recorded with log scales, compared call by call.

**Defaults for the step**, open to the user's correction:
- **GammaMixture**, on NormalMixture's pattern: `struct GammaMixture end`, interfaces
  `[:out, :switch, :a..., :b...]`, its own `GammaMixtureVMP`, and dependencies declaring the
  rates before the shapes (`:out => (q[:switch], q[:b...], q[:a...])`, `(:a, k) => (q[:out],
  q[:switch], q[:b][k])`, …), which is v6's `reverse(bs), reverse(as)` schedule without the
  reversal inside a group, which changes nothing (§3.24). The four rules keep v6's math: the
  switch rule renormalises after clamping, and `(:a, k)`, `(:b, k)` do not clamp, as in v6.
  The switch rule and the energy share `gamma_shape_rate_energy`, with no `score` call; the
  energy drops `init = 0.0` and takes any Gamma `q_b`, as the rules do. `INTEGER_SWITCH`
  guards as NormalMixture's, where v6 had none. Tests: v6's 7 cases, and v6's energy and
  GammaShapeLikelihood node items; its machinery items go with the machinery.
- **Mixture:** `struct Mixture end`, `[:out, :switch, :inputs...]`, `algorithm = MixtureBP`,
  dependencies over messages: `:out => (m[:switch], m[:inputs...])`, `:switch => (m[:out],
  m[:inputs...])`, `(:inputs, k) => (m[:out], m[:switch])`. The rules read incoming log scales
  as `getannotation(ann.m[:out], :logscale)` and so on, in place of `messages[i]`; a missing
  one is an error naming `LogScaleAnnotations`, where v6 raised a `KeyError`. The switch rule
  declares `ctx = (:product,)` and adds `last(ctx.product(m_out, m_inputs[k]))` to the two
  incoming log scales, then `logsumexp` and `softmax` (LogExpFunctions). v6's two
  `q_switch::PointMass` rules are not ported: only the deleted `RequireMarginal` path reached
  them. v6's math is kept and checked by hand-derived cases through
  `call_message_update_rule(...; ann = RuleAnnotations(m = …))`, and by the fixture. The
  `@define_factor_node` docstring's `MixtureVMP` example is renamed with the port.
- **The engine:** `rule_context` passes the `product` service, tested through a
  `MessageMapping`, as `ctx.rng` is.
- **INVENTORY:** `GammaMixtureNode`, `MixtureNode` and the `GaussianMixtureNode` alias are
  `delete`, as `NormalMixtureNode` is; the two mixture-rule notes point to the product service
  and `ann.m`.
- **Recorded, not done:** table cases taking incoming annotations; the engine calling
  `missing_services` (the follow-up table above).
- **Order:** GammaMixture; the engine's `product` service; Mixture, with the fixture and the
  docstring rename, closing the step. Each port gets comparison cases, and an issue tagging
  @Nimrais for any v6 mismatch.
- **Progress:**
  - *GammaMixture — done.* 4 message rules (6 with the `INTEGER_SWITCH` guards) and the
    energy, on NormalMixture's pattern under `GammaMixtureVMP`, the rates declared before the
    shapes. The switch rule and the energy share `gamma_shape_rate_energy`. v6's 7 cases and
    its energy and GammaShapeLikelihood node items are ported; the switch rule and the energy
    are checked in Float64 only, since ExponentialFamily's E[log Γ(a)] for a GammaShapeRate is
    a Float64 (a `0.5 *` literal, added to ExponentialFamily.jl#322) and a Categorical cannot
    go to BigFloat. Nothing differs from v6 (739 checks).
  - *The engine's `product` service — done.* `rule_context` passes `rule_product`, which
    multiplies with `GenericProd` and returns the product with `compute_logscale` of it alone.
    A test runs a rule declaring `ctx = (:product,)` through a `MessageMapping`, the log scale
    checked against ∫ N(x; 1, 2)² dx.
  - *Mixture — done, which closes step 8.* 3 message rules under `MixtureBP`, over messages,
    reading their inputs' log scales from `ann.m` (`incoming_logscale`, whose error names
    `LogScaleAnnotations`); the switch rule's evidence adds `ctx.product`'s log scale to the two
    incoming ones. No energy, so a free energy raises `RuleNotFoundError`. Hand-derived cases
    run through `call_message_update_rule(...; ann)`. The `mixture_bp` v6 fixture (log scales
    on; y observed through a NormalMeanVariance, since a data variable's message has no log
    scale in v6; no free energy) agrees exactly, log scales included, once three v6 calls the
    port does not make are set aside, each named in the test: the priors' deferred messages
    materialised a second time by v6's equality chain, and one `Mixture(:out)` from RxInfer
    computing z's marginal. TestUtils' fixture format gains `MixtureDistribution`. No
    rule-by-rule v6 comparison: v6's Mixture rules need annotated messages, which the oracle's
    calls do not carry; the fixture covers them.

### Step 9 brief — Close

**Scope, surveyed:** none of the three deliverables exists. There is no migration guide; all of
`docs/`'s ~30 pages describe v6 (`@node`, `@rule`, `meta`, `ManyOf`, functional dependencies,
`to_marginal`, …) and `@docs` names the engine no longer defines, `docs/make.jl` documents
`ReactiveMP` alone, and `make docs` exits 1, so CI's docs job fails; the engine's eight
`jldoctest`s run nowhere. `legacy/v6/` still holds the v6 engine and rule-system files
(`rule.jl`, `nodes/{nodes,dependencies,clusters}.jl`, `score/`, their tests), the rule
fallbacks (`fallbacks.jl`, `base`, unported), `StandaloneDistributionNode` (no inventory row),
helpers the unported nodes use (`helpers/algebra/common.jl`, `approximations/shared.jl`,
`fixes.jl`), and stale include lists.

**Decided (user, §3.36):**
- **The guide is the docs page `migration-guides/v6-to-v7.md`**, with no `MIGRATION.md`.
- **Only v7 code runs**: each pair's v6 side is plain code, its v7 side a doctest in the docs
  build. v6 is run nowhere.
- **Rewrite what is ported; drop the Phase 6 node pages**; keep the v5 → v6 guide as history.
- **`legacy/v6/`**: delete the v6 engine and rule-system files; mark the rule fallbacks and
  `StandaloneDistributionNode` not carried over (INVENTORY `delete`, with notes for the guide);
  keep the helpers the Phase 6 nodes use, with legacy/README saying whose they are.

**Defaults for the step**, open to the user's correction:
- **The docs build:** `docs/Project.toml` takes the lib packages through `[sources]`, and
  `docs/make.jl` documents `ReactiveMP` and the five lib packages, doctests on and `checkdocs`
  strict. The engine exports without docstrings (`tiny`, `huge`, `functionalform`,
  `getinterfaces`, `FactorBoundFreeEnergy`, `VariableBoundEntropy`, `DifferentialEntropy`) get
  them. Stale dependencies go (Optim, Parameters, Plots, StatsPlots, … if no page uses them).
  `make docs` runs the build again (`doc_init` then `docs/make.jl`), and CLAUDE.md and AGENTS.md
  say so. The root suite gains a `quality:doctests` item, as each lib package has, so the
  engine's doctests run with the tests too. The `.github/` workflows stay as they are until the
  first PR (Phase 7); the Delta package's missing LibTests job is recorded there.
- **The pages:** Introduction (the packages and how they fit); Concepts (factor graphs, message
  passing, reactive programming, the inference lifecycle, in the new terms); **Defining nodes
  and rules** (`@define_factor_node`, the rule macros, targets and groups, `FactorizedCluster`,
  annotations, the context services, `public_equivalent`); **Algorithms and dependencies** (the
  default scheme, a node's own algorithm, `DefaultAlgorithmExtension`, declared dependencies,
  the registry as introspection only); **Testing rules** (TestUtils' tables, verification, the
  comparison tools); the engine (factor nodes, `factornode` and `FactorNodeActivationOptions`,
  variables, messages, marginals, callbacks, postprocessors, annotations and log scales, the
  Bethe free energy, form constraints, helpers); the rule packages (Standard's nodes,
  Approximations, Delta); the migration guides (v6 → v7, new; v5 → v6, kept); Extra
  (contributing, exported methods). Removed: the Phase 6 node pages, `algebra.md` (its helpers
  are gone), `extensions.md` (both extensions dropped), the orphan `nodes/equality.md`.
- **The guide** follows PLAN § Migration guide: a preamble for an agent (what to read, what
  never to guess, how to verify, when to stop); mechanical pairs (`@node` →
  `@define_factor_node`, `@rule`/`@marginalrule`/`@average_energy` → the `@define_*` macros,
  `m_x`/`q_x` → `m[:x]`/`q[:x]`, `q_y_x` → `q[:y, :x]`, `ManyOf` → groups and `(:in, k)`,
  `Marginalisation` dropped, `meta` → an algorithm or a context service, `@logscale` →
  `annotate!`, `@call_rule` → a helper or `@call_message_update_rule`, NamedTuple marginals →
  `FactorizedCluster`, functional dependencies and RxInfer's `where { dependencies = … }` →
  declared dependencies and a node's algorithm, `to_marginal` → `public_equivalent`,
  `default_meta` → a rule's default, `DeltaMeta` → `DeltaApproximation`); the cases that cannot be
  translated mechanically, with stop-and-ask (raw `messages[i]`, graph objects built in rules,
  `meta` as mutable workspace); how to verify (the tables, `@verify_message_update_rule`, and
  `compare_with_reference` while a v6 is at hand); and the behaviour changes: the corrected v6
  errors (#669, #672–#680), the refused non-commuting `*` products, Mixture's missing energy,
  Delta's reduced method set, and the deleted exports with INVENTORY's notes.
- **Exit criteria:** "migrated per rule directory" and "hand-written cases done" are ticked
  with the close, as steps 1–8 did both; the `Require*` criterion's two pages are the
  dependencies page and the guide's section, written here, with Probit's and
  ContinuousTransition's algorithms left to their Phase 6 ports.
- **Order:** the legacy triage; the docs build back (infrastructure, docstrings, the root
  doctest item, and the pages' skeleton); the pages, a commit per section; the guide; the close
  (exit criteria ticked, CLAUDE.md's § Ongoing work to Phase 6).
- **Progress:**
  - *The legacy triage — done.* The v6 engine and rule-system files and their tests, the rule
    fallbacks, `StandaloneDistributionNode` and the stale include lists leave `legacy/v6/`;
    INVENTORY marks `NodeFunctionRuleFallback` and the fallback hook `delete`, with notes for
    the guide. legacy/README lists what stays and whose it is: the unported nodes, the
    approximations Phase 6 ports or deletes, their helpers, `fixes.jl` and the CVI extensions.
  - *The docs — done.* `docs/Project.toml` takes the engine and the lib packages through
    `[sources]`; `docs/make.jl` documents all six modules with `checkdocs = :exports` *(`:all` since the
    post-close review, as the brief said)* and runs
    every doctest and `@example`; `make docs` builds again. The engine's undocumented exports got
    docstrings, and `Message`'s and `Marginal`'s, which a comment had detached, attach again, so
    their doctests run for the first time; a root `quality:doctests` item runs the engine's
    doctests with the tests. New pages: *Defining nodes and rules*, *Algorithms and
    dependencies*, *Testing rules*, and a page per rule package; the engine pages are rewritten
    for the current API; removed: the unported nodes' pages, `rules.md`, `algebra.md`,
    `approximations.md`, `extensions.md`. No page carries historical remarks.
  - *The migration guide — done.* `migration-guides/v6-to-v7.md`: the agent preamble; pairs for
    nodes, message and marginal rules, energies, log scales, groups, `meta`, functional
    dependencies and the renames, their v7 side run by the build; what cannot be translated
    mechanically; how to verify a port; the behaviour that changed and what was removed.
  - *The close — done, which closes Phase 5.* The exit criteria are ticked, and the next action
    is Phase 6.

**Found while counting, to settle in the step that meets them:**
- `nodes/predefined/distribution/distribution.jl` (`StandaloneDistributionNode`) has no
  `INVENTORY.md` row and no engine handling. *(Settled in step 9: not carried over, and deleted
  from `legacy/v6/`.)*
- `GammaShapeLikelihood`, which GammaShapeRate's `a.jl` needs, is defined in v6's
  `gamma_mixture.jl`; it moves into Standard as a helper type in step 1. *(Settled in step 1:
  an exported type.)*
- dirichlet_collection, gamma, mixture and uninformative have no v6 rule tables; they get
  verification against the node definition or hand-derived cases. *(Gamma and Uninformative
  settled in step 3, with hand-derived cases.)*
- Standard grows dependencies as its nodes need them: LinearAlgebra and FastCholesky (the
  multivariate and matrix nodes), MatrixCorrectionTools (the correction strategy), DomainSets
  (NMV's `var.jl`, if its `HalfLine` log-pdf survives). *(DomainSets settled in step 1.)*
- The algebra helpers `diageye` (INVENTORY: `base`), `mul_trace` and `mul_inplace!` are needed by
  steps 5–7.
- The "~15 rules touching raw `messages[i]`/`marginals[i]`" are three in `standard`, the
  `Mixture` rules of step 8; the rest are DiscreteTransition's, Phase 6.

**Exit criteria**
- [x] ~~JuliaSyntax-based migration tool~~ — **dropped** (user, entry brief): ports by hand or
      agent, gated per directory as the brief says
- [x] migrated per rule directory, diffs reviewed per directory; each directory and its
      tests leave `legacy/v6/` in the same commit that ports it *(steps 1–8)*
- [x] canary passing: `NormalMixture((:m, k))` — indexed target + group + `where {N}` +
      aligned dependency *(done in Phase 4.5 step 3)*
- [x] **`MIGRATION.md` written *during* this phase, not after**, starting with the rules the 4.5
      ports found — the mechanical rules are
      discovered while porting, and reconstructing them later leaves gaps exactly where the
      work was fiddly. *(Its derivation from the transform tool went with the tool. Done in
      step 9 as the docs page `migration-guides/v6-to-v7.md`, from what steps 1–8 recorded;
      `DISCUSSION.md` §3.36.)*
- [x] every before/after pair in the guide is an executable doctest *(its v7 side, run by the
      docs build, locally until CI runs again; the v6 side is shown, never run, §3.36)*
- [x] guide covers the untranslatable cases explicitly (raw `messages[i]` indexing, rules
      constructing graph objects, `meta`-as-mutable-workspace) and tells the reader — human
      or agent — to stop and ask rather than guess
- [x] guide opens with a short preamble addressed to an agent: what to read, what never to
      guess, how to verify, when to stop
- [x] `docs/` rewritten for the new engine and rule system, and back in the build *(step 9;
      `make docs` builds and runs the doctests)*. The pages cover defining nodes
      and rules with the base macros, `factornode` and `FactorNodeActivationOptions`, algorithms
      and extensions, `bethe_free_energy`, and the registry as introspection only
      (`DISCUSSION.md` §3.23)
- [x] the engine distributes a `FactorizedCluster` to its members (step 2), first returned by
      the `PointMass` variants of NMV's and NMP's `(:out, :μ)` (step 1)
- [x] hand-written cases done: `mixture/switch.jl` and the `Mixture` rules reading raw
      `messages[i]` (step 8)
- [x] **`Require*FunctionalDependencies` are deleted, not ported** (user; `DISCUSSION.md` §3.21).
      *(Done for Phase 5 in step 9: the dependencies page and the guide's section are written.
      Probit's and ContinuousTransition's algorithms and Probit's initial message go with their
      ports, in § Phase 6.)*
      Probit and ContinuousTransition (Phase 6, `node:Probit` and `node:ContinuousTransition`) get
      their own algorithms from `ProbitMeta` and `CTMeta`, declaring their dependencies. Probit's
      self-dependency gets a default initial message declared on its node, separately from
      `dependencies`. Two pages are written with the port:
      - a documentation page on declaring dependencies, in the new terms only (the default
        scheme, a node's own algorithm, an extension for one model, initial messages);
      - a section of the v6 → v7 guide (`docs/src/migration-guides/v6-to-v7.md`, which is what
        `MIGRATION.md` became, §3.36) mapping the old types to those pieces, including RxInfer's
        `where { dependencies = … }`, which becomes choosing an algorithm for the node

### Post-close review — 2026-09-24

A review of the closed work at `cdd9f88e` found three defects and two gates `PLAN.md` promised
that nothing enforced; the consistency pass that followed resolved each:
1. **TestUtils ran a resolved rule under the requested algorithm**, not `rule_algorithm(spec,
   algorithm)`, so an inherited default rule saw the extension in its `algo` slot and its
   preallocation did too. Tables, verification and derivatives now pass the effective
   algorithm, as the engine does; `tables:algorithm-extension` and the extension cases in
   `derivatives:propagate` and `verification:correct-rules` pin it.
2. **The engine and `check_factorized_cluster` disagreed on a valid partition**: the engine
   required the blocks, concatenated, to reproduce the cluster's order. It now asks what the
   public check asks, so blocks may come in any order and need not be contiguous
   (`engine:factorized-cluster:arguments`).
3. **NormalMixture and GammaMixture lost v6's constructor checks**, so unequal groups were
   accepted and the switch rule's `zip` dropped a component. `@define_factor_node` now declares
   `matched_groups`, `min_group_length` and `factorisation` (user: a declaration, not a check
   inferred from the dependencies), the engine's `factornode` enforces them, and both
   mixtures declare all three, restoring v6's checks.
4. **The docs build checked exported docstrings only** (`checkdocs = :exports`, where the step 9
   brief and `PLAN.md` § Documentation say every docstring). It is `:all` now: the 65 missing
   docstrings are on their pages, or were genuinely internal and are comments.
5. **The registry-backed coverage gate was never run.** Standard and Delta now run
   `check_rule_coverage` after an unfiltered suite; a direct `call_*` counts as a table case
   does (§3.39). It found 50 of Standard's rules unselected: 39 tested by direct calls, and 11
   without a test, which have one now. Delta had none.

Beyond the findings, the pass reconciled the documents with the state of the repository: the
status table, the not-done table, what `legacy/v6/` holds, `MIGRATION.md` as the guide's docs
page, Phase 6's node list, and `INVENTORY.md`'s annotation, scoring and trait rows, which are
the engine's (§3.37, where log scales are also marked experimental until Phase 7 decides).

---

## Phase 6 — Approximations and node packages

`lib/DeltaMessagePassingRules` exists since Phase 4.5 case (d), with the Delta node, its
algorithm and its Unscented rules; the items below that concern Delta add to it.

### Entry brief — 2026-09-24

**Scope, counted** (`legacy/v6/src/rules/` and `nodes/predefined/`, line-start macros; the
destinations are `INVENTORY.md`'s). 13 nodes in 10 packages: **123 message rules, 22 marginal
rules and 19 average energies**, and DiscreteTransition's 3 hand-written `rule`, 3 `marginalrule`
and 2 `score` methods. Delta adds its Linearization rules (4 message, 1 marginal) and the
projection extension's (2 and 2). Of the 123, GCV defines 4 on NormalMeanVariance and
NormalMeanPrecision; of the 22, 2 are MvNormalMeanPrecision's for BIFM and 4 are GCV's on the
Standard normals.

| package | nodes | rules / marginals / energies | v6 meta → | functional dependencies | extra dependencies | v6 tests |
|---|---|---|---|---|---|---|
| Autoregressive | AR, ConjugateAR | 13 / 3 / 3 | `ARMeta(order, ARsafe \| ARunsafe)`, required → an algorithm | — | LazyArrays, StatsFuns | ~41 table cases, 35 node tests |
| SoftDot | SoftDot | 8 / 1 / 2 | none (builds an `ARMeta` inside) | — | StatsFuns | ~63 cases |
| GaussianCoupling | GaussianCoupling | 2 / 1 / 1 | none | — | — | ~17 cases, 15 node tests |
| Probit | Probit | 7 / 1 / 1 | `ProbitMeta(p = 32)` → an algorithm | `in = NormalMeanPrecision(0, 100)`: its own edge, an initial message | StatsFuns, cubature | ~37 cases |
| GCV | GCV | 14 / 5 / 2 | `GCVMetadata(GaussHermiteCubature(20))` → an algorithm | — | StatsFuns, cubature, `approximate_meancov` | ~110 tests |
| ContinuousTransition | ContinuousTransition | 8 / 1 / 2 | `CTMeta(f)`, required → an algorithm | `RequireMarginal(a = nothing)` → declared | ForwardDiff, StatsFuns | ~15 cases |
| Polya | BinomialPolya, MultinomialPolya | 4 / 0 / 2 | `BinomialPolyaMeta(n, rng)`, `MultinomialPolyaMeta(points)` | the β and ψ rules read their own edge | PolyaGammaHybridSamplers (GPL-3), SpecialFunctions, LogExpFunctions, cubature | ~7 cases |
| BIFM | BIFM, BIFMHelper | 6 / 1 + 2 / 1 | `mutable BIFMMeta`, required, a cache the rules write | BIFMHelper's overridden | — | ~21 cases, 18 node tests |
| Flow | Flow | 8 / 4 / 0 | `FlowMeta(model, Linearization() \| Unscented(d))` → an algorithm | — | TupleTools | ~54 cases, ~466 model tests |
| DiscreteTransition | DiscreteTransition | 53 / 3 / 5, + 8 hand-written | ignored | — | Tullio | ~159 cases |
| Delta (exists) | Linearization, `CVIProjection` | 4 / 1, ext 2 / 2 | `DeltaApproximation` | — | ForwardDiff (via Approximations); ExponentialFamilyProjection (weakdep) | 7 + 3 + 4 cases, ext ~13 |

**Decided (user):**
- **The monorepo stays through Phases 6 and 7; the split into repositories is Phase 8**, with
  registration, when compat bounds and CI are set anyway (`DISCUSSION.md` §3.40). `PLAN.md`
  had it at Phase 6.
- **SoftDot is independent of AR.** Its `y` rule called AR's, and it used AR's `ar_slice` and
  `add_transition`; its package reimplements what it needs and is tested on its own (§3.40).
- **Gauss–Hermite cubature and `approximate_meancov` go to `MessagePassingRulesApproximations`**,
  which gains FastGaussQuadrature: Pólya, Probit and GCV all use them, where `INVENTORY.md` had
  Pólya alone. The part over means and covariances moves; the methods that take a distribution
  stay with the node packages, so the package still depends on no distribution package (§3.40).
- **The algebra helpers go to `StandardMessagePassingRules`**: `mul_trace`, `rank1update`,
  `negate_inplace!` and `mul_inplace!`, used by AR, SoftDot, ContinuousTransition and
  DiscreteTransition, beside its `diageye` (§3.40).
- **Order**: numerics first, then by dependency, with a brief per step (below).
- Already decided: a package per node (§3.30); Probit's and ContinuousTransition's own
  algorithms, and Probit's default initial message (§3.21); the Pólya package carries the GPL-3
  sampler (`PLAN.md` § Licensing); BIFM and `CVIProjection` are `pure = false` (`PLAN.md`
  § Purity); CVI, Laplace, importance sampling, Gauss–Laguerre, spherical-radial cubature and
  the Optimisers extension are deleted, not ported; Delta's methods are `Unscented` and
  `Linearization`, with `CVIProjection` as an extension, and #11's two follow-ups (§3.16).

**Defaults**, open to the user's correction:
- **Packages** are `<Node>MessagePassingRules` (`AutoregressiveMessagePassingRules`,
  `PolyaMessagePassingRules`, …), on Delta's template: `[sources]` to their siblings, the test
  runner with its coverage gate, `quality_tests.jl` (Aqua, closure, doctests, `quality:rules`), a
  docs page under *Rule packages*, a `make test-<node>` target, and entries in `docs/make.jl`,
  `docs/Project.toml`, the root `[extras]` and CLAUDE.md. A package depends on Standard when its
  rules use Standard's helpers or nodes.
- **Gates**, as in Phase 5: v6's tables; a case per rule against 6.5.0 through
  `compare_with_reference`, in a `compare_<node>.jl` per package, disagreements declared with
  their reasons; `@verify_message_update_rule` where the inputs allow; `check_rules` and
  `check_rule_ambiguities` finding nothing; the coverage gate at zero; and an engine fixture
  per node with a small model, recorded from RxInfer 5.5.2. Each directory and its tests leave
  `legacy/v6/` in the commit that ports it.
- **Delta's v6 comparison** extends `V6Oracle` to build a v6 `DeltaFnNode` with a `DeltaMeta`,
  and the new side passes `RuleContext(node = …)`; Delta joins the comparison environment.
- **Rules on another package's node.** GCV's 8 rules on NormalMeanVariance and
  NormalMeanPrecision take GCV's own `ExponentialLinearQuadratic`, so they stay in its package;
  v6 never tested them, and they get tests. BIFM's two MvNormalMeanPrecision marginals over a
  `TerminalProdArgument` move to Standard, whose types they are.
- **v6 errors found while counting**, corrected, declared in the comparisons *(with no issue, by the
  user's decision in the step 6 brief)*:
  BinomialPolya's energy computes its Monte Carlo term and overwrites it on the next line;
  ContinuousTransition's `default_meta` is defined on `CTMeta` instead of the node, and its
  energy takes the output dimension as half the joint's, which holds only when both are equal
  *(the step 5 brief found both energies wrong in three terms, this one included)*;
  DiscreteTransition's four five-interface `:T2` rules read `m_T2`, their own edge, so they never
  match, its belief-propagation `:T3` rule with a Dirichlet `q_a` normalises over `dims = 1` unlike
  its siblings, and its fast-path marginal on `Val{(:a)}` is `Val{:a}` and unreachable; BIFM's
  MvNormalMeanPrecision marginals call `getdist`, defined nowhere, so they failed in 6.5.0.
- **Engine and base work the ports meet**, each designed in its step brief:
  - a default initial message declared on a node, applied at activation to an edge that has
    none (Probit's `in`); RxInfer's side stays Phase 7;
  - the Pólya β and ψ rules and Probit read the message on their own edge, which the declared
    dependencies already express;
  - DiscreteTransition's `T...` group and its hand-written rules, which read inputs by name:
    ported by hand, stopping to ask where no mechanical form exists;
  - BIFM's rules run in an order and share its cache: `pure = false`, and its step checks the
    schedule the engine gives it;
  - table cases taking incoming annotations (`ann.m`) if a second node needs them.
- **`CVIProjection`** is a weakdep extension of the Delta package on ExponentialFamilyProjection;
  `cvilinearize`, which it takes from the deleted `cvi.jl`, moves into it; its generator becomes
  `ctx.rng`, and its mutable proposal keeps it `pure = false`.
- **`legacy/v6/src/fixes.jl`** is dropped: its only users were `laplace.jl` and `cvi.jl`.
  Linearization never takes a Hessian.

**Order of work**, a brief per step, a commit per node, `PHASES.md` and `CHANGELOG.md` in each:
1. **Numerics and deletions.** `Linearization` into Approximations, with ForwardDiff: its
   `approximate(::Linearization, g, x̂)` returns `(A, b)`, where Unscented's returns `(m, V)`, so
   the Delta side wraps each. Gauss–Hermite cubature, `approximate_meancov` and
   FastGaussQuadrature into Approximations, with v6's buffer-reuse test (#633). The algebra
   helpers into Standard. The deletions, and `Optim` confirmed gone.
2. **Delta**: the Linearization rules, the `CVIProjection` extension, #11's follow-ups (a
   guarded positional constructor; an error naming the package and the method to switch to),
   the Delta v6 comparison, a `delta_linearization` fixture, and the guide's Delta section.
3. **The small nodes**: GaussianCoupling; Probit, with the initial message; GCV, with
   `ExponentialLinearQuadratic` and its product.
4. **The autoregressive family**: AR and ConjugateAR, with the companion matrix and the standard
   basis vector (their `*` methods narrowed first, for the Aqua ambiguities) and `ARMeta` as an
   algorithm; ConjugateAR's `@call_rule`s become helpers. Then SoftDot, on its own.
5. **ContinuousTransition**: `CTMeta` as an algorithm, `RequireMarginal` as declared dependencies.
6. **Pólya**: both nodes, the GPL-3 sampler, BinomialPolya's generator as `ctx.rng`.
7. **BIFM**: BIFM and BIFMHelper, whose overridden dependencies become declared ones; the
   Standard marginals; `pure = false`.
8. **Flow**: `PermutationMatrix`'s `*` methods narrowed first; `FlowMeta` as an algorithm over
   `Linearization` or `Unscented`; the layers and the model.
9. **DiscreteTransition**: Tullio, the `T...` group, the fast paths and the hand-written rules.
10. **Close**: `legacy/` deleted, the guide's node sections complete, the exit criteria ticked.

**Found while counting, to settle in the step that meets them:**
- the companion matrix is used: AR builds it with `as_companion_matrix` at seven sites.
  `INVENTORY.md` had it deleted, from a search for the type's name, and the last consistency
  pass repeated it; it goes to the AR package;
- Probit's marginal rule and GCV's normal extension have no v6 tests, and BinomialPolya's `y`
  test item is misnamed `rules:BinomialPolya:beta`;
- ContinuousTransition imports LazyArrays and does not use it;
- Flow's layers draw from the global generator when constructed, and its unscented rules read
  `Unscented(d)`'s weights, which Approximations does not export;
- BIFMHelper's average energy is the entropy of `q_in`, a bookkeeping device;
- ConjugateAR's single-interface `w` marginal is reached only by its tests;
- the `w` message of ConjugateAR can be improper for order ≥ 3, as v6 documents;
- DiscreteTransition declares its traits by hand, with no `@node`.

### Step 1 brief — numerics and deletions

**Scope, surveyed** (`legacy/v6/src/approximations/`, `helpers/algebra/common.jl`, their tests):
- **`Linearization`** (`linearization.jl`, 142 lines): the method type, `approximate(::Linearization,
  g, x̂)` returning the local linear map `(A, b)`, and `local_linearization` with its methods for
  scalar or vector inputs and outputs, one input or several (through `shared.jl`'s
  `__splitjoin`/`__as_vec`, already ported). Its distribution-level `approximate` over normals
  and its `is_delta_node_compatible` belong to the Delta package (step 2). 5 v6 checks.
- **Gauss–Hermite cubature** (`gausshermite.jl`, 90 lines): `GaussHermiteCubature(p)`,
  `ghcubature`, `getweights`, `getpoints`, univariate and multivariate; the multivariate points
  are one reused buffer, by design (#633), with a test that pins it.
- **`approximate_meancov`** (`approximations.jl`): the mean and variance of `g(x)` for a normal
  `x` given by its moments, by any method with `getweights`/`getpoints`; GCV calls it.
- **The algebra helpers** (`common.jl`): `negate_inplace!`, `mul_inplace!`, `rank1update` and
  `mul_trace`, used by AR, SoftDot, ContinuousTransition and DiscreteTransition, with their v6
  tests (`common_tests.jl`).
- **Deleted, not ported:** `cvi.jl` (all but `cvilinearize`, which step 2 moves into the
  `CVIProjection` extension), `laplace.jl`, `importance.jl`, `gausslaguerre.jl`,
  `sphericalradial.jl`, `optimizers.jl`/`optimizers/adam.jl`, `ext/ReactiveMPOptimisersExt`,
  `fixes.jl`, and their tests; from `approximations.jl`, `approximate_kernel_expectation`, whose
  only user was Laplace; from `common.jl`, `powerset`, `v_a_vT` and `rank1update!`, which nothing
  uses, and `isonehot` and `diageye`, which Standard already has.

**Decided (user, the entry brief):** Linearization, the cubature and `approximate_meancov` go to
`MessagePassingRulesApproximations`, which takes ForwardDiff and FastGaussQuadrature and still
depends on no distribution package and not on the base package; the algebra helpers go to
Standard.

**Defaults for the step**, open to the user's correction:
- **The numerical API is carried over**, as the exit criterion says: `approximate(::Linearization,
  g, x̂)` keeps returning `(A, b)` and `approximate(::Unscented, f, means, covs)` `(m, V)`; the
  Delta package wraps each in step 2. Exported: `Linearization`, `local_linearization`,
  `GaussHermiteCubature`, `ghcubature`, `getweights`, `getpoints`, `approximate_meancov`.
- **`approximate_meancov` takes moments only**, `(method, g, m, v)` and `(method, g, m, P)`;
  v6's `(method, g, distribution)` forwarding method stays with GCV, its caller.
- **v6's errors, corrected:** `approximation_name` and `approximation_short_name` of a
  `GaussHermiteCubature` read a field `p` it does not have, so both threw; they report the
  number of points. `gausshermite.jl` loaded Distributions for nothing.
- **The algebra helpers** are Standard's, unexported, beside `diageye`, and documented on its
  page under *Helpers*: node packages take them qualified.
- **Tests:** v6's Linearization checks with `@inferred`, and new ones for its other shapes (a
  vector output from a vector input, three inputs); the cubature's exactness on polynomials of
  degree `2p - 1`, univariate and multivariate, `approximate_meancov` against closed forms, and
  v6's buffer test without its spherical-radial half; v6's algebra tests. Each suite's quality
  items keep checking the dependency closure.
- **v6 comparison:** `compare_approximations.jl` gains Linearization and Gauss–Hermite cases
  against 6.5.0, exact up to `1e-12`.
- **`Optim`** is confirmed gone: nothing outside the pinned 6.5.0 environment's manifest names it.

**Order:** Linearization, with its comparison; the cubature and `approximate_meancov`, with
theirs; the algebra helpers in Standard; the deletions. A commit each.

**Progress:**
- *Linearization — done.* `Linearization`, `approximate(::Linearization, g, x̂)` and
  `local_linearization` in `MessagePassingRulesApproximations`, with ForwardDiff; v6's five
  checks and cases for a scalar and a vector function of a vector, a Jacobian and three mixed
  inputs; `compare_approximations.jl` agrees with 6.5.0 on six cases (28 checks in all). The
  distribution-level `approximate` over normals is step 2's, in the Delta package.
- *Gauss–Hermite cubature and `approximate_meancov` — done.* `GaussHermiteCubature`,
  `ghcubature`, `getweights`, `getpoints` and `approximate_meancov` over moments, with
  FastGaussQuadrature; exactness on polynomials of degree `2p - 1`, `approximate_meancov`
  against closed forms, v6's buffer tests, and 14 more checks against 6.5.0 (42 in all). v6's
  display names, which threw, report the number of points. The distribution package stays out
  of the closure.
- *The algebra helpers — done.* `negate_inplace!`, `mul_inplace!`, `rank1update` and
  `mul_trace` in Standard, unexported and documented, with v6's tests (`helpers:algebra`);
  `rank1update!` is folded into `rank1update`'s generic path. `common.jl` and its tests have left
  `legacy/v6/`.
- *The deletions — done, which closes step 1.* Out of `legacy/v6/`: the ported
  `approximations.jl`, `shared.jl`, `linearization.jl` and `gausshermite.jl`; CVI, Laplace,
  importance sampling, Gauss–Laguerre, spherical-radial cubature and the optimisers; the Delta
  CVI rules and layout; `ext/ReactiveMPOptimisersExt`; `fixes.jl`; and their tests. What stays for
  step 2 is `cvi_projection.jl`, `ext/ReactiveMPProjectionExt`, Delta's Linearization rules and its
  default layout. `CVIProjection` needs `cvilinearize` from the deleted `cvi.jl`, two methods,
  `cvilinearize(v::AbstractVector) = v` and `cvilinearize(m::AbstractMatrix) = eachcol(m)`,
  which step 2 writes into the extension. `Optim` appears only in prose and in the pinned 6.5.0
  environment's manifest.

### Step 2 brief — Delta

**Scope, surveyed** (`legacy/v6/src/rules/delta/linearization/`, `nodes/predefined/delta/`,
`src/approximations/cvi_projection.jl`, `ext/ReactiveMPProjectionExt/`, their tests):
- **Linearization**: 4 message rules and 1 marginal rule, the same four targets and the same
  shapes as the Unscented rules the package has since Phase 4.5 case (d). Only the forward
  statistics differ: the unscented transform's `(μ, Σ, C)` there, the linear map's
  `(Aμ + b, AΣAᵀ, ΣAᵀ)` here; the division towards an input without an inverse is the same code.
  v6's tables: 7 cases towards `out`, 8 towards `in`, 4 for the joint, all with promotion checks.
- **`CVIProjection`**: the method type (an RNG, sample counts, projection parameters, a mutable
  proposal and a sampling strategy, `FullSampling` or `MeanBased`), and in the extension on
  ExponentialFamilyProjection 3: `DivisionOf`, a lazy quotient with its products, and 2 message
  rules and 2 marginal rules. Its layout is `:out` from `m[:out]`, `q[:out]` and `q[(:in,)]`, the
  one Delta layout reading its own edge; towards an input and the joint as the default one.
  It ignores a given inverse, with a warning. v6's tests: about 13 rule cases, JET checks.
- **#11's two follow-ups**: the positional `DeltaApproximation(method, inverse)` skips the
  compatibility guard, and the guard's error names neither the method to use nor the package.
- **No Delta rule-by-rule comparison exists**; `V6Oracle` calls rules with `node = nothing`, and
  v6's Delta rules read the function from a real `DeltaFnNode`.

**Decided (user, entry brief):** `CVIProjection` is a weakdep extension of the Delta package on
ExponentialFamilyProjection; its generator is `ctx.rng`, and its mutable proposal makes it
`pure = false`; the method set is `{Unscented, Linearization}` and `CVIProjection` with the
extension; `cvilinearize` moves into the extension.

**Defaults for the step**, open to the user's correction:
- **One set of rules for both Gaussian methods.** The four rules are typed on
  `DeltaApproximation{<:Union{Unscented, Linearization}}`, and the method enters through two
  helpers: `approximate_normal`, gaining a Linearization method (v6's `approximate` over normals,
  through `joint_mean_cov` instead of a `JointNormal`), and a new `forward_statistics(method, f,
  μs, Σs)`. The Unscented tables stay, and v6's Linearization tables are added with their
  promotion checks.
- **#11:** the guard moves into an inner constructor, so no constructor skips it, and its error
  names the methods the node takes and, for `CVIProjection` without the extension, the package
  to load: `is_delta_node_compatible` stays the opt-in, and a method may add a hint.
- **`CVIProjection`** is defined in the Delta package (INVENTORY: `node:Delta`) without its rules;
  `ext/DeltaMessagePassingRulesProjectionExt` holds `DivisionOf`, the rules, the compatibility
  opt-in and `cvilinearize`. Its algorithm declares the layout above as dependencies, with no
  engine change: a target may read its own edge, and a deterministic node's `q[:out]` is the
  variable's marginal. The `RNG` field goes, the rules declaring `ctx = (:rng,)`; the proposal
  stays, as the algorithm's state. v6's rule tests are ported with a `StableRNG`, the JET checks
  dropped. An engine test checks that a `DivisionOf` message forms the right marginal, since the
  engine's product, not the rule, divides it out.
- **The Delta v6 comparison**, `compare_delta.jl`: `V6Oracle` builds a v6 `DeltaFnNode` with a
  `DeltaMeta` for each case, and the new side passes the function through `RuleContext(node =
  …)`; Delta joins the comparison environment. Cases per rule, for both Gaussian methods;
  `CVIProjection` is sampled, and is checked by its tests instead.
- **A `delta_linearization` engine fixture**, recorded from RxInfer 5.5.2 as `delta_unscented`
  was, and its `@testitem`.
- **The guide**: the Delta section says the node takes `Unscented`, `Linearization` and, with
  ExponentialFamilyProjection, `CVIProjection`, and that the other v6 methods are gone, as the
  breaking entry the exit criterion asks for.

**Order:** Linearization's rules and tables; #11; the comparison; the fixture; `CVIProjection`;
the guide, which closes the step. A commit each.

**Progress:**
- *Linearization — done.* The four rules are typed on both Gaussian methods (`rules/gaussian.jl`,
  renamed from `unscented.jl`); `approximate_normal` gains Linearization's method and
  `forward_statistics` takes the method's forward moments. Both go through the inputs'
  `JointNormal`, not `joint_mean_cov`: for one scalar input it keeps the mean a scalar, which the
  output type follows. v6's tables (7 cases towards `out`, 8 towards an input, 4 joints) pass with
  their promotion checks, the test functions keeping v6's integer constants.
- *#11 — done.* The guard is `DeltaApproximation`'s inner constructor, so the positional one no
  longer skips it; the error names the node's methods, the package for `CVIProjection`, and the
  method's `delta_method_hint`, which an extension's method can add (`delta:algorithm`).
- *The Delta v6 comparison — done.* `compat/v6-comparison/compare_delta.jl` runs every Delta
  rule against 6.5.0's for Linearization and two Unscented parameter sets: towards `out`,
  towards an input with and without an inverse, and the joint, 102 checks, all agreeing.
  `V6Oracle` takes the v6 node a rule reads, and builds a `DeltaFnNode` as v6's `@call_rule`
  did (`v6_delta_node`); Delta joins the comparison environment.
- *The fixture — done.* `delta_linearization`, `x ~ NMV(0.5, 1)`, `z := x³ - x` by Linearization,
  `y ~ NMV(z, 0.1)` observed, recorded from RxInfer 5.5.2, agrees call by call, posteriors and
  free energy included (`engine:fixture:delta_linearization`). *Found:* with two random inputs,
  v6 computes the first input's prior message once per subscriber, the same value three times
  where the engine computes it once, so a call-by-call comparison cannot match it; the values
  agree. Recorded for Phase 7's checks on scheduling, and the fixture has one input.
- *`CVIProjection` — done.* The method type, its sampling strategies and the proposal container
  are in the Delta package, without v6's `rng` field; its dependencies are declared (`:out` from
  `m[:out]`, `q[:out]` and `q[(:in,)]`), with no engine change. The extension
  `DeltaMessagePassingRulesProjectionExt`, on ExponentialFamilyProjection 3, holds `DivisionOf`
  and its products, `cvilinearize`, the projection families and the three rules, which draw
  from `ctx.rng`; the joint rule, which replaces the proposal, is `pure = false`. A known
  inverse is ignored with a warning, as in v6. v6's rule tests are ported with a StableRNG; the
  Delta suite's coverage gate covers the extension's module; `engine:function-node:cvi-projection`
  checks, on a linear function, that the product at the variable divides the `DivisionOf` back
  out. *Found, both in v6:*
  - the joint over several inputs updated each input from the same samples of the previous
    proposal, so for `x * y` observed at 2 the two inputs flipped between modes of opposite sign
    on successive calls, and v6's test that the proposal converges held for its generator's
    stream only. *Corrected (user):* the inputs are projected in turn, each against the others'
    latest projections, its samples redrawn from its projection before the next input; the KL
    divergence then falls from about 21 to 6.5 in two calls and the proposal settles on one
    mode, and v6's convergence test is back, stricter;
  - `optimize_parameters` and `create_density_function` were reached only by their own tests
    and are not ported, nor are v6's JET checks and its benchmark.
  `legacy/v6/` holds nothing of Delta any more.
- *The guide — done, which closes step 2.* The v6 → v7 guide's Delta entry names the three
  methods, the ones gone with no replacement, the error that refuses them, and `CVIProjection`'s
  generator and state, the breaking entry the exit criterion asks for; its `meta` section maps
  `DeltaMeta` with Linearization and an inverse.

### Step 3 brief — the small nodes

**Scope, surveyed** (`legacy/v6/src/nodes/predefined/{gaussian_coupling,probit,gcv}.jl`, their
rule directories and tests):
- **GaussianCoupling** (`out`, `in`, `a`): 2 message rules and 1 marginal rule, all with a
  point-mass `q_a` and messages on `out` and `in`, and its energy; improper messages by design.
  About 17 table cases and 15 node tests, one of which solves a linear system by belief
  propagation.
- **Probit** (`out`, `in`): 7 message rules, 1 marginal rule and its energy. `ProbitMeta(p = 32)`
  sets only the energy's Gauss–Hermite points. The rules towards `in` are expectation propagation:
  they read the message on their own edge, which v6 seeded with `NormalMeanPrecision(0, 100)` at
  activation (`RequireMessageFunctionalDependencies(in = …)`). Two rules forwarded a point-mass
  `q_out` to the `m_out` rules with `@call_rule`. The marginal rule has no v6 test. About 37 cases.
- **GCV** (`y`, `x`, `z`, `κ`, `ω`): 10 message rules, 1 marginal rule and 2 energies, variational
  under any factorisation of `(y, x)`; `GCVMetadata(GaussHermiteCubature(20))` names the
  cubature of `ExponentialLinearQuadratic`, a distribution GCV owns, with its product against
  normals. `gaussian_extension.jl` adds 4 message rules and 4 marginal rules to NormalMeanVariance
  and NormalMeanPrecision for an `ExponentialLinearQuadratic` message, each an `@call_rule` into the
  normal's own rule, and v6 never tested them. About 110 tests, with a test module of its own.

**Decided (user):**
- **A node may declare a default initial message** (§3.21, confirmed): `@define_factor_node(…,
  initial_messages = [:in => NormalMeanPrecision(0.0, 100.0)])`, kept on the `NodeSpec`. At
  activation the engine seeds the node's inbound message on that interface only when nothing was
  set there, so a user's initialisation wins. Probit declares v6's.
- **Probit's algorithm reads messages only**, expectation propagation: `:out => (m[:in],)`,
  `:in => (m[:out], m[:in])`. It is v6's behaviour for an observed `out`, the usual `y ~ Probit(x)`.
  v6's rules taking `q_out` are ported and tested by direct calls; the engine does not select them
  under this algorithm.

**Defaults for the step**, open to the user's correction:
- **Packages** `GaussianCouplingMessagePassingRules`, `ProbitMessagePassingRules` and
  `GCVMessagePassingRules`, on Delta's template (the entry brief), each with a Makefile target, a
  docs page, a `compare_<node>.jl` against 6.5.0 and a place in CLAUDE.md. GCV depends on Standard,
  whose normals its extension rules serve; Probit and GCV on Approximations, for the cubature.
- **Algorithms**: `ProbitEP(; p = 32)` succeeds `ProbitMeta`, and `GCVApproximation(; method =
  GaussHermiteCubature(20))` succeeds `GCVMetadata`, the node's own. GCV declares no dependencies:
  the default scheme follows its factorisation, as v6's did.
- **Rules that called rules** call helpers: Probit's forwarding pair calls the `m_out` rules'
  helper; GCV's normal extensions reduce the `ExponentialLinearQuadratic` to its moments and call
  the helpers of Standard's normal rules, taken qualified, or their formulas where Standard has
  none.
- **Tests**: v6's tables and node tests, with type-promotion checks where the rules allow them;
  hand-derived cases for Probit's marginal and GCV's normal extensions, which v6 did not test.
- **Engine fixtures**: a Probit model whose loop needs the default initial message, and a
  GaussianCoupling chain; GCV if a small model runs in 6.5.0.

**Order:** the `initial_messages` declaration and the engine seeding it; GaussianCoupling; Probit;
GCV; the guide's entries for the three, which close the step. A commit each.

**Progress:**
- *Initial messages — done.* `initial_messages` on `@define_factor_node`, validated at
  declaration (an interface, not a group, each once), on the `NodeSpec` and its display; the
  engine's `activate!` seeds each on the node's inbound message, marked initial, where
  `getrecent` finds none. Tests: the declaration in the base package, and the seeding, a user's
  value winning and a node without any, in the engine (`nodes_tests.jl`). The dependencies page
  has a section on it.
- *GaussianCoupling — done.* `lib/GaussianCouplingMessagePassingRules` (`make
  test-gaussian-coupling`): the node, its two improper messages, its joint and its energy, with
  v6's tables and node tests, type promotion included, and v6's GaBP solver of `A x = b` by direct
  calls; the cross-checks against NormalMeanPrecision are written as their closed forms, since the
  package does not depend on Standard. `compare_gaussian_coupling.jl`: 136 checks, all agreeing.
  A `gaussian_coupling` fixture, `x ~ NMP(0.5, 2)`, `c ~ GaussianCoupling(x, 0.5)`, `y ~ NMV(c, 1)`
  observed, agrees in values; v6 computes `x`'s prior message once per subscriber, and orders
  the node's two messages otherwise within an iteration, so `compare_engine_trajectory` gains
  `collapse_repeats`, declaring the repeats, and the test declares both. The docs page is
  *Rule packages › GaussianCoupling*.
- *Probit — done.* `lib/ProbitMessagePassingRules` (`make test-probit`): the node, with
  `initial_messages = [:in => NormalMeanPrecision(0.0, 100.0)]`, and its algorithm `ProbitEP(; p =
  32)`, declaring `:out => (m[:in],)` and `:in => (m[:out], m[:in])`. *Found:* `check_rules`
  requires a rule under an algorithm to consume exactly its declared inputs, so v6's rules that
  read `q_out`, `q_in`, or `m_out` alone cannot be `ProbitEP`'s. They are `DefaultAlgorithm`'s,
  the default scheme choosing between messages and marginals by the factorisation, as v6's
  non-EP behaviour did; only the `(q_out, m_in)` forwarding rule, which no algorithm reaches, is
  not ported. The energy is under both, with 32 points under the default. *Corrected:* the energy
  took `log(normcdf(x))`, which underflows at far cubature points: Inf, or NaN for a point-mass
  output, for a wide `q(in)` and 100 points; it takes `normlogcdf`, declared in the comparison.
  v6's tables pass with their promotion checks; the marginal, untested in v6, is checked against
  the cavity times the EP message and against the closed form. `compare_probit.jl`: 205 checks,
  the underflow cases declared. `probit_ep`, two Probit outputs of one weight, agrees with v6
  call by call, which only the declared initial message makes possible.
- *GCV — done.* `lib/GCVMessagePassingRules` (`make test-gcv`): the node, `GCVApproximation(;
  method = GaussHermiteCubature(20))` for `GCVMetadata`, no declared dependencies (the default
  scheme follows the factorisation), `ExponentialLinearQuadratic` with its moments by cubature and
  its product with a normal, the 10 message rules, the joint and the two energies; and the 8 rules
  that let NormalMeanVariance and NormalMeanPrecision take an `ExponentialLinearQuadratic` on
  `out`, their `@call_rule`s written out with Standard's helpers (`variational_variance`,
  `coupled_precision`, `promoted_cluster`), and tested for the first time. The v6 tests were
  ported by a subagent and reviewed: 461 checks. Aqua found an ambiguity v6 also had, between the
  default constructor and the all-`Integer` one, which is narrowed. *Found:* the product with a
  normal centres its cubature on the normal, where the doubly exponential factor is poorly
  resolved, so its variance can be 3.5e-2 off; v6's test asserted 1e-2 for its generator's draws
  only, and the port's is 5e-2 with that reason, the rule kept as v6's. `compare_gcv.jl`: 85
  checks, the #669 cases of the normal extension declared, as in Standard. A `gcv_meanfield`
  fixture, with `y` observed through a narrow normal so that v6's energy applies, agrees with v6
  call by call, free energy included; `ExponentialLinearQuadratic` gains `params` for the fixture
  encoder, and the recorder gives v6's the same.
- *The guide — done, which closes step 3.* The v6 → v7 guide's *Node packages* section maps each
  node to its package and its `meta` to the node's algorithm, with Probit's initial message and
  GCV's distribution; its behaviour changes include Probit's finite energy.

### Step 4 brief — the autoregressive family

**Scope, surveyed** (`legacy/v6/src/nodes/predefined/{autoregressive,conjugate_autoregressive,softdot}.jl`,
their rule directories, `helpers/algebra/{companion_matrix,standard_basis_vector}.jl`, the tests):
- **AR** (`y` alias `out`, `x`, `θ`, `γ`): 8 message rules, 1 marginal rule `(:y, :x)`, 2 energies,
  in mean-field and structured `q(y, x)` pairs. `ARMeta{F}(order, ARsafe | ARunsafe)` is required,
  with no default: the variate form (Univariate forces order 1), the order, and whether the
  covariance is regularised. Its own matrix types, `ARPrecisionMatrix` and `ARTransitionMatrix`,
  broadcast into dense ones; the companion matrix of `θ` (`as_companion_matrix`, seven sites) and
  a `StandardBasisVector` (`ar_unit`) are v6's algebra types, whose `*` methods v6 counted at 75 and
  85 Aqua ambiguities. `LazyArrays` served one index vector in the energy. About 41 table cases,
  30 node tests.
- **ConjugateAR** (`y`, `x`, `w`): 5 message rules, 2 marginal rules, 1 energy; `w` is the joint
  `(θ, γ)` as an `MvNormalGamma`. Its `x` and `y` rules and its energy map `q(w)` to effective
  `(q_θ, q_γ)` and call AR's (`@call_rule`, `score`); its `w` message can be improper for order ≥ 3,
  as v6 documents; one marginal rule is over the single interface `w`. Its tests compare with AR
  and with a Bayesian linear regression.
- **SoftDot** (`y`, `θ` alias `theta`, `x`, `γ` alias `gamma`), declared as `softdot`: 8 message rules,
  1 marginal rule, 2 energies. Its `y` rule called AR's with an `ARMeta` it built, and it used AR's
  `ar_slice` and `add_transition`. About 63 cases, and tests pinning ReactiveMP.jl#615.

**Decided (user, entry brief):** AR and ConjugateAR share a package, with the companion matrix and
the standard basis vector; SoftDot has its own, independent of AR's, reimplementing what it needs
and tested on its own; the algebra helpers (`mul_trace`, `rank1update`, …) are Standard's.

**Defaults for the step**, open to the user's correction:
- **Packages** `AutoregressiveMessagePassingRules` (AR, ConjugateAR) and `SoftDotMessagePassingRules`,
  on the Phase 6 template, depending on Standard for its helpers and MvNormalGamma's algebra.
- **The algorithm**: `ARMeta` becomes `ARVMP(form, order, stype)`, v6's constructor and its order
  check kept, with `ARsafe()` and `ARunsafe()`. Like Delta, the nodes declare no algorithm of their
  own: v6 had no default meta, so under `DefaultAlgorithm` no rule is found, and the error says so.
  ConjugateAR takes the same algorithm. SoftDot has none, as in v6.
- **Rules that called rules** share helpers: ConjugateAR's `x` and `y` rules and its energy call the
  AR helpers with the effective marginals; SoftDot's `y` rule writes out AR's `y` formula.
- **ConjugateAR's `w` marginal** is not ported: a one-member cluster of a single interface is its
  marginal, which the engine forms from the messages, and only v6's tests reached it; its algebra
  is tested through the `w` message's product with a prior.
- **The algebra types** move to the AR package unexported, their `*` methods narrowed until Aqua
  finds no ambiguity, which every lib package checks; `LazyArrays` goes, the index vector built.
- **Tests**: v6's tables and node tests, ported by a subagent against the port's API and reviewed,
  as GCV's were; type-promotion checks where v6 asked for them.
- **v6 comparisons** `compare_autoregressive.jl` and `compare_softdot.jl`, and **engine fixtures**
  with free energy: an AR(2) process under `q(y, x) q(θ) q(γ)`, and a SoftDot regression.

**Order:** the AR package (algebra types first, then AR, then ConjugateAR); SoftDot; the guide's
entries, which close the step. A commit each.

**Progress:**
- *AR and ConjugateAR — done.* `lib/AutoregressiveMessagePassingRules` (`make
  test-autoregressive`), ported by a subagent and reviewed: the two nodes, `ARVMP`, `ARsafe` and
  `ARunsafe`, the companion matrix and the standard basis vector unexported. Aqua finds no
  ambiguity, since the algebra types' `*` methods are narrowed and v6's `broadcast!` overloads on
  `ARPrecisionMatrix` and `ARTransitionMatrix` became `add_precision!` and `add_transition!`.
  `LazyArrays` is gone. The rules share `ar_towards_y`, `ar_towards_x`, `ar_joint` and `ar_energy`;
  ConjugateAR calls them with `conjugatear_effective_marginals(q_w)`. 12 854 checks.
  *Corrected:* v6's `ARunsafe` joint had a Schur complement of the wrong sign and one transpose too
  many. Checked in 6.5.0 itself for a univariate AR(1), with `m_y = N(1, 1)`, `m_x = N(0, 1)` and
  `θ = γ = 1`: v6's covariance is `[0.4 0.2; 0.2 0.6]`, where `ARsafe` and the exact inverse give
  `[0.667 0.333; 0.333 0.667]`. For a multivariate AR it inverted the singular companion matrix and
  threw. The port uses the Kalman gain, pinned against a closed-form joint. `compare_autoregressive.jl`:
  188 checks over AR(1), AR(2) and AR(3), both modes, with the `ARunsafe` joint declared a
  correction. A `y` message from `q(x)` carries each package's own lazy `ARTransitionMatrix`, so
  the script compares entries. Two fixtures agree with v6 call by call, free energy included:
  `ar_meanfield`, an AR(1) under mean-field, and `ar2_structured`, the brief's AR(2) under
  `q(x0, x) q(θ) q(γ)`.
- *SoftDot — done.* `lib/SoftDotMessagePassingRules` (`make test-softdot`), ported by a subagent and
  reviewed. It does not depend on AR's package. `split_y_x` replaces v6's `ar_slice`, and `y_from_x`
  is AR's `y` message reduced to its first component, which reads only the companion matrix's first
  row, `⟨θ⟩ᵀ`, so the matrix is never formed. 928 checks, ReactiveMP.jl#615's tests included.
  `compare_softdot.jl`: 66 checks at orders 1 to 3, all agreeing. A `softdot_regression` fixture
  agrees with v6.
  *Found:* under mean-field the engine updates the posterior subscribed last first, so the harness
  subscribes `γ` before `θ` to reproduce v6's schedule, where γ updates last. `ar_meanfield` needs
  the same.
- The ported files and their tests have left `legacy/v6/`, with `helpers/algebra/{companion_matrix,
  standard_basis_vector}.jl`.
- *The guide — done, which closes step 4.* The v6 → v7 guide's *Node packages* table gains AR and
  ConjugateAR (`ARMeta` → `ARVMP`) and SoftDot, with a note that no rule runs without `ARVMP`.
  Its behaviour changes gain `ARunsafe`'s corrected joint.

### Step 5 brief — ContinuousTransition

**Scope, surveyed** (`legacy/v6/src/nodes/predefined/continuous_transition.jl`, `rules/continuous_transition/`,
the tests):
- **ContinuousTransition** (`y`, `x`, `a`, `W`; alias `CTransition`) is `y ~ N(A(a) x, W⁻¹)`, with
  `A = f(a)` an `dy × dx` matrix. It has:
  - 8 message rules, one mean-field and one structured `q(y, x)` for each of `a`, `W`, `x` and `y`;
  - the marginal rule `(:y, :x)`;
  - 2 energies.
- **`CTMeta(f)`**, alias `ContinuousTransitionMeta`, is required and has no default. For a
  nonlinear `f`, `ctcompanion_matrix` linearises it at `m_a + std(a)` and evaluates the result at
  `m_a`. The variance terms use the Jacobians `Fᵢ` of `f`'s rows at `m_a`, by ForwardDiff. For a
  linear `f`, `A = f(m_a)` exactly.
- **Dependencies:** v6's `RequireMarginalFunctionalDependencies(a = nothing)`. The rule towards
  `a` reads `q(a)`, its expansion point; every other target follows the factorisation. v6
  documents both factorisations: mean-field and `q(y, x) q(a) q(W)`.
- **Tests:** about 49 cases in 5 rule files. The node tests pin both energies against a "manual
  calculation", and check `ctcompanion_matrix`.

**Found while surveying, checked in 6.5.0** against a Monte Carlo estimate (400 000 draws) and
the closed form. The closed form agrees with Monte Carlo to 0.5% in six cases: `(dy, dx)` =
`(2, 2)`, `(1, 2)`, `(2, 3)`, each mean-field and structured.
- **Both energies are wrong**, by 1.1 to 2.4 nats in those cases. Three terms reproduce v6's value
  exactly:
  - `E[log det W]` is not halved;
  - the `log2π` term's dimension is `div(ndims(q_y), 2) = dy/2` under mean-field, always wrong,
    and `div(dx + dy, 2)` under the structured factorisation, right only when `dx = dy`. The entry
    brief knew only this case;
  - the `q(a)` covariance term `Σᵢⱼ Wᵢⱼ tr(Fᵢᵀ E[xxᵀ] Fⱼ Va)` takes `E[xxᵀ]` as `I + m_x m_xᵀ`,
    not `V_x + m_x m_xᵀ`.
- **v6's node test pins its own formula.** Its inputs (zero means, identity covariances,
  `Wishart(3, I)`) have `V_x = I`, which hides the third term. The exact value is 13.415 (Monte
  Carlo 13.386), where the test asserts 12.992 structured and 12.077 mean-field.
- **The rules are consistent with the correct expectation**; only the energies are wrong. The
  `W` rule's `WishartFast(dy + 2, Δ)` is the likelihood with the halved log-determinant.
- `default_meta` is defined on `CTMeta`'s type instead of the node's, so its error never fired.
- `ctcompanion_matrix` adds to `f(a0)`'s result in place.
- `LazyArrays` is imported and unused.

**Decided (user):**
- **Auxiliary inputs, 2026-09-24, `DISCUSSION.md` §3.41.** A dependency declaration can add inputs
  to the default scheme's for a target, so one algorithm serves both factorisations. Two
  algorithms by factorisation, and dropping mean-field, were the alternatives.
- **From the entry brief:** a package of its own, and `CTMeta` becomes the node's algorithm.

**Defaults for the step**, open to the user's correction:
- **The declaration.** `default` is a word of the dependency vocabulary: `:a => (default, q[:a])`
  is the default scheme's inputs for `a`, plus `q(a)`, and `:y => (default,)` is the default
  scheme alone.
  - A target left undeclared under a declared spec stays an error, so the node declares all four.
  - `TargetDependencies` records that a target extends the default scheme.
  - `validate_dependencies` refuses `default` twice, or an auxiliary input that is a cluster.
- **The engine.** `declared_dependencies` takes `default_dependencies` and adds each auxiliary input.
  - Each goes among the marginals in interface order, where v6's `insertafter` put it; the
    subscription order is the VMP schedule (§3.24).
  - An auxiliary input the default already has is not added twice.
  - An auxiliary marginal is consumed and never scored, as `PLAN.md` § Open questions 9 decided.
  - The cycle `m(→a) → q(a) → m(→a)` is broken by `combineLatest(…, PushNew())`, as in v6:
    the message is recomputed once every input has refreshed. An engine test pins it with a toy
    node, and the partition test checks that `q(a)` is not scored twice.
- **`check_rules`.** A rule for a target that extends the default scheme must read every auxiliary
  input. Its other inputs depend on the factorisation and are not checked; the error says so.
- **Documentation:** a section in *Algorithms and dependencies*, and the guide's mapping of
  `RequireMarginalFunctionalDependencies` to `default` plus the marginal.
- **The package.** `ContinuousTransitionMessagePassingRules`, on the Phase 6 template.
  - It depends on Standard for `rank1update`, `negate_inplace!` and `mul_trace`, and on
    ForwardDiff and FastCholesky.
  - `CTMeta(f)` becomes `CTVMP(f)`, and `CTransition` is kept. The node declares no algorithm,
    as AR and Delta, so under `DefaultAlgorithm` no rule is found: this replaces v6's dead
    `default_meta` error.
  - `ctcompanion_matrix` does not modify `f`'s result, and `LazyArrays` goes.
- **The energies are corrected and declared.** The three terms are fixed, and v6's node-test
  values are replaced by the closed form, with a Monte Carlo test at `dx ≠ dy`. An issue is opened.
- **Tests:** v6's tables and node tests, ported by a subagent against the port's API and reviewed,
  with type-promotion checks where v6 asked for them.
- **v6 comparison.** `compare_continuous_transition.jl`:
  - covers a linear `f` (`reshape`) and a nonlinear one (a rotation), with `dx = dy` and
    `dx ≠ dy`;
  - covers every rule, the joint and both energies;
  - declares the energies as corrections.
- **Engine fixtures,** with free energy: a two-dimensional rotation model under `q(y, x) q(a) q(W)`,
  and a mean-field one if a small model runs in 6.5.0. The free energy is v6's error, declared;
  the posteriors and every rule call must agree.

**Order:**
1. the auxiliary inputs, base and engine together, failing tests first;
2. the package;
3. the comparison and the fixtures;
4. the guide's entries, which close the step.

A commit each.

**Progress:**
- *Auxiliary inputs — done.* In the base package, `default` joins the dependency vocabulary:
  - `TargetDependencies` records it, keeping a three-argument constructor for the rest;
  - `extends_default_scheme(declaration, target)` reads it;
  - the display shows `default, q[:a]`.
  - The macro refuses `default` twice. `validate_dependencies` refuses beside it anything but a
    single interface's message or marginal, so groups are left out as well as clusters, until a
    node needs one.
  - `check_rules` requires a rule for such a target to read the added inputs.

  In the engine, `extended_default_dependencies` takes `default_dependencies` and places each added
  input in interface order. A marginal goes after the clusters whose first member precedes it, as
  v6's `insertafter` did, and it is the variable's, never a cluster, so free energy does not see
  it. An input the default already has is skipped.

  Tests, failing first:
  - the declaration, its errors and display, and a `check_rules` case, in the base;
  - the placement under mean-field and `q(y, x)`, the skipping, and a message added among messages;
  - a toy `y ~ N(a x, 1/W)` whose rule towards `a` reads `q(a)`. Over four iterations it makes
    one call per target per iteration, the cycle broken by `PushNew()`, and the node keeps its
    four clusters.

  Docs: *Algorithms and dependencies › Extending the default scheme*, and the guide's mapping of
  `RequireMarginalFunctionalDependencies`.
- *The package, its comparison and its fixtures — done.* `lib/ContinuousTransitionMessagePassingRules`
  (`make test-continuous-transition`):
  - the node and `CTVMP(f)`, with the declaration `:a => (default, q[:a])` and `default` for the
    other targets;
  - the 8 message rules, the joint and the two energies;
  - one Jacobian of `vec ∘ f` in place of v6's one per row, and `ct_matrix` building a new matrix
    where v6 added to `f`'s result in place;
  - `LazyArrays` is gone.

  v6's tables were ported by a subagent and reviewed; with the energy, offset and quality tests
  the suite is 519 checks.

  *Corrected, the energies:* the closed form, pinned at v6's node-test inputs (13.415, where v6
  asserted 12.992 and 12.077), against Monte Carlo at `(dy, dx)` = `(1, 2)`, `(2, 3)` and `(3, 2)`
  within 0.5%, and for Float32 through Standard's `gaussian_energy`.

  *Found, and corrected by the user's decision (`DISCUSSION.md` §3.42):* v6's rules towards `a` and
  `W` took each row of A as linear through the origin, so for an affine or nonlinear `f` they
  dropped the offset `f(m_a) - J m_a` that the other rules and the energy keep. A Monte Carlo check
  with `f = I + [0 a; 0 0]` gave the `W` rule's Δ as `[2.44 -0.54; -0.54 0.62]` against
  `[3.19 -0.18; -0.18 2.46]`. The two rules now use the same linearisation, the offset
  `O = Ā - [(Fᵢ m_a)ᵀ]` entering the rule towards `a` as `E[x (y - O x)ᵀ]`, and the rule towards
  `W` taking the energy's `E[(y - A x)(y - A x)ᵀ]`. `O` is zero for `reshape`, where nothing
  changes. The test is a reparametrisation: `B + reshape(a)` on `q(y, x)` equals `reshape` on the
  joint of `(y - B x, x)`, rule by rule and in the energy, to 1e-9. v6's two rotation cases for `a`
  and `W` moved to a test of their own, asserting a proper result.

  `compare_continuous_transition.jl`: 132 checks over `reshape` at four sizes, a rotation and an
  affine `f`, with the energies and the nonlinear `a` and `W` rules declared as corrections.

  Two fixtures, `ct_structured` and `ct_meanfield`, record a linear two-dimensional chain under
  `q(x0, x) q(a) q(W)` and under mean-field. Every posterior and every rule call agrees with v6, with
  `a` subscribed first. The free energy is set aside, since v6's was wrong.
  *Found:* the free energy must still be subscribed to. Under `q(x0, x)` only it reads `q(x0)`, so
  without it the message towards `x0` is never computed and the traces differ by one call per
  iteration. The corrected free energy is not monotone under v6's schedule either: it rises by
  0.002 in the structured fixture's last iteration, though not under another subscription order.
  So v6's rising free energy was no evidence of its energy errors, and the fixture asserts no
  monotonicity.

  The ported files have left `legacy/v6/`.
- *The guide — done, which closes step 5.* The v6 → v7 guide's *Node packages* table gains
  ContinuousTransition (`CTMeta` → `CTVMP`), with a note that it declares no algorithm and still
  reads `q(a)`, which the model initialises. Its behaviour changes gain the corrected energies and
  the corrected rules for a nonlinear `f`. The two commits for the package, its comparison and its
  fixtures became one, as in steps 3 and 4.

### Step 6 brief — the Pólya nodes

**Scope, surveyed** (`legacy/v6/src/nodes/predefined/{binomial,multinomial}_polya.jl`, their rule
directories, the tests):
- **BinomialPolya** (`y`, `x`, `n`, `β`) is a binomial regression through the logistic,
  `y ~ Binomial(n, σ(xᵀβ))`, with Pólya-Gamma augmentation. It has 2 message rules and 1 energy.
  - The rule towards `β` reads `q(y)`, `q(x)`, `q(n)` and `m(β)`, its own edge's message: `ω` is the
    Pólya-Gamma mean at `xᵀ mean(m_β)`.
  - The rule towards `y` reads `q(x)`, `q(n)` and `q(β)`.
  - `BinomialPolyaMeta(n_samples, rng)` is optional, and its default is `nothing`: the rules use
    the means, or with a meta average over samples of `β` drawn from its `rng`.
- **MultinomialPolya** (`x`, `N`, `ψ`) is a multinomial through logistic stick-breaking. It has
  2 message rules and 1 energy.
  - The rule towards `ψ` reads `q(x)`, `q(N)` and `m(ψ)`, its own edge.
  - The rule towards `x` reads `q(N)` and `q(ψ)`.
  - `MultinomialPolyaMeta(ncubaturepoints)` defaults to 21 points, for the energy's Gauss–Hermite
    cubature.
  - v6 exported `logistic_stick_breaking` and `compose_Nks`.
- **Dependencies:** v6 declared none, so its default scheme never gave `m(β)` or `m(ψ)`. Each model
  passed `where { dependencies = RequireMessageFunctionalDependencies(β = …) }` with an initial
  message, as RxInfer's binomial and multinomial regression tests do (§3.21).
- **Libraries:** PolyaGammaHybridSamplers (GPL-3), SpecialFunctions, LogExpFunctions, and the
  cubature now in Approximations.
- **Tests:** about 7 cases, and node tests of both energies. The Monte Carlo paths are tested
  against the mean paths with tolerances.

**Found while surveying, checked in 6.5.0** against exact expectations: a one-dimensional
quadrature of `E[softplus(ψ)]`, and exact enumeration over `q(x)`.
- **BinomialPolya's energy is always the plug-in `softplus(E[ψ])`.** Its Monte Carlo term is
  computed and then overwritten, so asking for 1 000 samples returns the plug-in too. For one
  normal `q(β)` the plug-in is 1.069 against the exact 1.534: softplus is convex, so the plug-in is
  biased low.
- **MultinomialPolya's energy is wrong for a Multinomial `q(x)` with N > 1:** 4.559 against 3.157 at
  N = 3, and 3.950 against 2.742 at N = 5. For N = 1, and for a point-mass `q(x)`, it matches.
  - `E[log C]`, the expected log multinomial coefficient, is `log N! - Σ_k E[log x_k!]`. v6
    computes `Σ_k E[log x_k!] - log N` instead, the sign flipped and `log N` for `log N!`.
  - v6's node test asserts −101.72 for `q(x) = Multinomial(100, …)`. That is impossible: the
    average energy of a discrete `x` is at least zero.

**Decided (user):**
- **BinomialPolya's energy by Gauss–Hermite, 2026-09-24.** `ψ = xᵀβ` is normal under a normal `q(β)`,
  so `E[softplus(ψ)]` is a one-dimensional integral over `N(xᵀm, xᵀVx)`: deterministic, and as
  MultinomialPolya's energy already computes it. The energy then takes neither samples nor a
  generator. Fixing only the overwrite was the alternative.
- **No issues for the v6 errors that the ports correct, 2026-09-24,** this step's or earlier ones'.
  The entry brief's "each with an issue" is dropped: each correction is declared in its comparison
  and recorded here and in the changelog.
- **From the entry brief:** one package for both nodes, carrying the GPL-3 dependency, with
  BinomialPolya's generator taken from `ctx.rng`.

**Defaults for the step**, open to the user's correction:
- **The package:** `PolyaMessagePassingRules`, on the Phase 6 template. It depends on
  PolyaGammaHybridSamplers, SpecialFunctions, LogExpFunctions and Approximations, and the README
  and the docs page say it is GPL-3 through the first.
- **The algorithms:**
  - `BinomialPolyaMeta` becomes the node's own algorithm, `BinomialPolyaApproximation(; samples =
    nothing)`. `nothing` is v6's mean path and the node's default; a number of samples draws `β`
    from `ctx.rng`, so the rules stay pure (the generator is the caller's).
  - `MultinomialPolyaMeta(points)` becomes `MultinomialPolyaApproximation(; points = 21)`, also the
    node's default.
  - The approximation names follow `GCVApproximation`, since they choose how, not what.
- **Dependencies, by §3.41's extension:**
  - BinomialPolya: `:β => (default, m[:β])`, and `default` for `y`, `x` and `n`;
  - MultinomialPolya: `:ψ => (default, m[:ψ])`, and `default` for `x` and `N`.

  The model no longer passes dependencies. No initial message is declared, since the dimension of
  `β` or `ψ` is the model's: a model initialises `m(β)` as it did, and the guide says so.
- **The energies are corrected and declared:**
  - BinomialPolya's by Gauss–Hermite over `ψ`, 32 points like Probit's.
  - MultinomialPolya's Multinomial branch as `log N!`-based, `-log N! + Σ_k E[log x_k!]`, with the
    marginals `Binomial(N, π_k)`.
  - v6's node-test values are replaced by the closed form and by enumeration, the −101.72 case
    included.
- **The helpers** `logistic_stick_breaking` and `compose_Nks` stay exported, as in v6, since
  models use the first to read a posterior; `INVENTORY.md` has them in the package already.
- **Tests:** v6's cases, ported by a subagent and reviewed. The Monte Carlo rule paths draw from a
  `StableRNG` passed as `ctx.rng`, and their tolerances stay v6's.
- **v6 comparison, `compare_polya.jl`:**
  - both nodes' rules on the mean paths, which are deterministic;
  - the Monte Carlo paths are compared as distributions, not draws, since the port's generator is
    the caller's;
  - both energies, with the corrections declared.
- **Engine fixtures:** a binomial regression and a multinomial one, small, on the default algorithms
  and with observed counts. The multinomial energy for observed counts is right in v6, so its
  free energy must agree; the binomial's is a declared correction. Both models initialise `m(β)` or
  `m(ψ)`. The harness can only set initial marginals today, so it gains initial messages, set as
  RxInfer 5.5.2 sets `μ(β)`; how that is done is read in 5.5.2 before writing it.

**Order:**
1. the harness's initial messages, if the fixtures need them;
2. the package, BinomialPolya then MultinomialPolya;
3. the comparison and the fixtures;
4. the guide's entries, which close the step.

A commit for the package with its comparison and fixtures, as in steps 3–5, and one for the guide.

**Progress:**
- *The harness's initial messages — done.* `H.run` takes `initial_messages` (variable =>
  distribution), set on every message out of the variable after its initial marginal, as RxInfer
  5.5.2 sets `μ(x)`. v6's per-node `RequireMessageFunctionalDependencies(β = …)` seeds each node's
  inbound message on `β`; `μ(β)` seeds the same messages and one more, on the prior's edge, which
  the prior's rule never reads, so the two are equivalent for these models.
- *The package, its comparison and its fixtures — done.* `lib/PolyaMessagePassingRules`
  (`make test-polya`), its README and docs page stating GPL-3:
  - both nodes, with `BinomialPolyaApproximation{S}` and `MultinomialPolyaApproximation` as their
    algorithms;
  - the declarations `:β => (default, m[:β])` and `:ψ => (default, m[:ψ])`;
  - BinomialPolya's rules in two forms, `{Nothing}`, the mean path, and `{Int}` with
    `ctx = (:rng,)`, so that only sampling needs the generator;
  - MultinomialPolya's Categorical `q(x)` branch dropped. v6's energy read the size of its scalar
    mean first and threw, so the branch was unreachable.

  v6's tables were ported by a subagent and reviewed: 108 checks with the energy and quality
  tests, and the coverage gate.

  *Found:* a node's `algorithm = T` with a parametric `T` binds its inline dependencies and its
  default rules to the default instance's type, `BinomialPolyaApproximation{Nothing}`, so the
  sampling variant had neither. BinomialPolya declares both against `T` with `@define_dependencies`
  and `algorithm = T`; `CLAUDE.md` records it as a gotcha.

  *Corrected, the energies,* as the brief says:
  - BinomialPolya's by 32-point Gauss–Hermite, against a fine quadrature to 1e-8;
  - MultinomialPolya's Multinomial branch against enumeration to 1e-6, and v6's 23.76 for observed
    counts kept, since it was right.

  *Found by the subagent and corrected, a v6 error the tables did not reach:* the sampling path
  took `eachcol` of a univariate `β`'s draws, one column of `k`, so `dot(x, column)` threw for more
  than one sample; v6 tested only one. Draws are samples now, with a failing test first.

  `compare_polya.jl`: 60 checks.
  - The mean paths agree exactly.
  - The Monte Carlo paths are compared as distributions, 20 000 samples each on different
    generators, and agree within 2%.
  - The energies agree for a point-mass `q(β)`, and for observed counts or a single trial; the
    rest are declared corrections.

  Two fixtures. `multinomial_regression` agrees with v6 in everything, free energy included.
  `binomial_regression` agrees in every posterior and rule call, and its corrected free energy is
  above v6's at every iteration, as Jensen's inequality requires of the plug-in.

  The ported files have left `legacy/v6/`.
- *The guide — done, which closes step 6.* The v6 → v7 guide's *Node packages* table gains both
  Pólya nodes, with their metas as algorithms. A note says to drop the per-model
  `RequireMessageFunctionalDependencies` and initialise the message instead, and that the package
  is GPL-3. Its behaviour changes gain the corrected energies.

### Step 7 brief — BIFM, and scratch space for rules

**Scope, surveyed** (`legacy/v6/src/nodes/predefined/bifm{,_helper}.jl`, `rules/bifm{,_helper}/`,
`rules/mv_normal_mean_precision/marginals.jl`, the tests; the RxInferExamples notebook
*RTS vs BIFM Smoothing*):
- **BIFM** (`out`, `in`, `zprev`, `znext`) is a whole time slice of a linear state-space model,
  `znext = A zprev + B in`, `out = C znext`, for backward-information-filter forward-marginal
  smoothing. It is deterministic. It has 4 message rules and the marginal `(:in, :zprev, :znext)`,
  and its messages towards `in`, `out` and `znext` are `TerminalProdArgument`s: marginals, not
  messages.
- **`BIFMMeta(A, B, C)` is mutable and required.**
  - The rule towards `zprev` writes `H`, `BHBt`, `ξz`, `Λz`, `ξ̃z` and `Λ̃z` into it; the rules
    towards `out` and `znext` write `μu` and `Σu`.
  - The rules towards `in`, `out` and `znext` read them back, so results depend on the update
    order, and v6's docstring asks for a subscription order.
  - A second constructor, `BIFMMeta(A, B, C, μu, Σu)`, fixes the input's statistics, which the
    `znext` rule then overwrites.
- **BIFMHelper** (`out`, `in`) switches from the backward pass to the forward one. It has 2
  message rules: towards `in` its `m_out`, and towards `out` `TerminalProdArgument(q_in)`. Its
  energy is `entropy(q_in)`. v6 overrode its functional dependencies: `in` reads `m(out)`, `out`
  the other clusters' marginals.
- **MvNormalMeanPrecision** has two marginal rules for a `TerminalProdArgument` on `out` or `μ`.
  They call an undefined `getdist` (its field is `.argument`).
- **Tests:** v6's rule tests (about 21 cases) and node tests of the meta's getters and setters.
  Neither RxInfer nor RxInferExamples tests BIFM beyond the example.

**Found while surveying, checked in 6.5.0:**
- The example's model runs, and its posteriors of `z[2:end]` and `u` equal the equivalent RTS
  smoother's exactly, on a 2-dimensional state with 4 observations.
- Its free energy fails: "Failed to compute node bound free energy component. The result is `Inf`".

**Decided (user, 2026-09-24, `DISCUSSION.md` §3.44):**
- **BIFM is stateless.** Where v6 read the meta's cache, each rule recomputes from its inputs,
  reading its own edge's message through `default` plus `m[...]` (§3.41):
  - the rule towards `in` reads `m(in)`, for `μu` and `Σu`;
  - the rules towards `out` and `znext` read `m(out)` and `m(znext)`, for `Λz`, `BHBt` and `ξ̃z`;
  - the marginal already reads all four messages.

  The algorithm is immutable, and the rules are pure, independent of order, and safe to share
  across nodes.
- **Rules get scratch space.** A rule may declare `scratch = (args) -> …`, working memory, and take a
  `scratch` slot in its body. The engine owns it, one per outbound stream, built from the first
  call's inputs and reused. It is **write-before-read**: it carries nothing between calls, and the
  engine may keep it, drop it or rebuild it, so the rule stays pure. It is separate from
  `inplace`, and the two combine. Nothing is shared across a node's rules: that would be v6's
  cache again, with its ordering hazard.
- **BIFM's free energy is out of scope,** as in v6, where it failed: computing the free energy of
  a BIFM model raises an error naming the node instead of an `Inf`, and a correct Bethe free
  energy for BIFM is left as a follow-up.

**Defaults for the step**, open to the user's correction:
- **Scratch in the base:**
  - `BODY_SLOTS` becomes `(output, scratch, algo, ctx, args, ann)`;
  - `scratch` is a keyword on the message and marginal rule macros, taking the slots
    `(algo, ctx, args)` as `preallocate` does;
  - `RuleSpec` gains the function;
  - `execute_rule` takes the store the engine keeps, with `nothing` asking for a fresh one, as
    `output` does now.
- **The engine:** each `MessageMapping` and `MarginalMapping` keeps a scratch slot, built at the
  first call and reused while the same rule runs on that stream. An edge keeps its dimensions
  within a graph; a rule whose shapes can vary checks them itself.
- **TestUtils checks the contract.** A table case of a rule with scratch runs twice: once with a
  fresh scratch, once with a reused one poisoned with NaN (`poison!` dispatching on arrays, tuples
  and named tuples). The two results must agree, so a rule that reads before writing fails its own
  tables.
- **An allocation test** (`:alloc`) shows the reuse, and the documentation gets a section in
  *Defining nodes and rules*.
- **BIFM:** the package `BIFMMessagePassingRules` holds BIFM and BIFMHelper.
  - `BIFMMeta(A, B, C)` becomes the immutable algorithm `BIFMSmoother(A, B, C)`, required, which
    the node declares no default for.
  - The five-argument constructor goes, since `m(in)` gives the input's statistics.
  - Its rules are the first to use scratch, for their intermediates.
  - BIFMHelper's dependencies are declared: `:in => (m[:out],)` and `:out => (default,)`.
- **MvNormalMeanPrecision's two `TerminalProdArgument` marginals** move to Standard, with `.argument`
  for `getdist`, returning a `FactorizedCluster`.
- **Tests:** v6's rule tables, ported by a subagent and reviewed. The meta's getter and setter
  tests go with the meta.
- **v6 comparison, `compare_bifm.jl`:** v6's rules need the cache in a state the `zprev` rule
  leaves, so the comparison calls v6's `zprev` rule first on the same inputs, then each forward
  rule. The port must agree call by call.
- **Engine fixture, `bifm_smoother`:** the example's model, small, without free energy. Posteriors
  and rule calls are compared with v6; the trace order may differ, since the forward rules also
  wait for their own edge's message, and `:within_iteration` is declared if it does.
  - An engine test checks the posteriors against the equivalent RTS model built from Standard's
    nodes, independently of v6.
  - A test checks the free-energy error.
- **The fixture encoder** may need `TerminalProdArgument`, the form of BIFM's posteriors.

**Order:**
1. scratch, in the base, the engine and TestUtils, failing tests first;
2. the MvNormalMeanPrecision marginals in Standard;
3. the BIFM package;
4. the comparison and the fixture;
5. the guide's entries, which close the step.

Scratch is its own commit, the BIFM package with its comparison and fixture another, and the
guide a third.

**Progress:**
- *Scratch — done.*
  - In the base, `scratch` is a keyword of the message and marginal macros, not of energies, and
    its slots are `preallocate`'s. The macro requires the keyword and the body slot together.
    `RuleSpec` carries the builder, `rule_scratch` builds a scratch, and `execute_rule` gains an
    eight-argument form taking one; the seven-argument form builds a fresh one, as interactive
    calls do.
  - In the engine, `src/scratch.jl` defines `ScratchSlot` and `scratch_for!`, and each
    `MessageMapping` and `MarginalMapping` keeps a slot, rebuilt only when another rule runs on
    the stream.
  - In TestUtils, a table case of a rule with scratch runs again on a reused scratch, before and
    after `poison!`, into fresh outputs, and must agree.

  Tests, failing first:
  - base: declaration and calls, in-place with scratch, a group index, a marginal rule, the
    display, the macro errors, and the reuse saving the scratch's allocation;
  - TestUtils: a rule that accumulates into its scratch fails its table, and one that writes first
    passes;
  - engine: three calls on one stream build one scratch, and another stream its own.

  Docs: *Defining nodes and rules › Scratch*, the testing page and the mappings' API.
- *BIFM — done.* `lib/BIFMMessagePassingRules` (`make test-bifm`).
  - BIFM, BIFMHelper, and `BIFMSmoother(A, B, C)` in place of `BIFMMeta`: immutable, with checked
    sizes and promoted element types.
  - The declaration adds each forward target's own message: `:out => (default, m[:out])`, and so
    for `in` and `znext`.
  - One helper, `bifm_backward!`, computes `ξz`, `Λz`, `H`, `BHBt`, `ξ̃z` and `Λ̃z` in v6's order of
    operations into the rule's scratch, the first rules to use it. Each of the four rules calls it
    on its own messages, so they are pure and independent of order.
  - BIFMHelper's dependencies are declared, `:in => (m[:out],)` and `:out => (default,)`.

  *Found, and not ported:* v6's marginal `(:in, :zprev, :znext)` returned the joint of `in` and
  `zprev` only, and only the free energy reached it. The cluster's marginal rule now raises
  `BIFMFreeEnergyError`, as BIFMHelper's energy does, the error the free energy meets. For the same
  reason MvNormalMeanPrecision's two `TerminalProdArgument` marginals are not ported: only a BIFM
  model's free energy reached them. This is a deviation from the brief, which moved them to
  Standard. The free-energy follow-up would restore them.

  *Found, and replaced:* v6's tables towards `in`, `out` and `znext` filled the meta's cache with
  arbitrary numbers inconsistent with the messages, so a stateless rule cannot reproduce them;
  only the `zprev` table ported as it was. A subagent wrote the tables anew from 6.5.0, on a fresh
  meta after v6's `zprev` and `znext` rules on the same messages, and they are pasted as literals:
  - 23 BIFM cases, with a 3-dimensional state and non-square `B`;
  - BIFMHelper's two tables, ported;
  - 419 checks with the scratch reruns and the quality tests.

  `compare_bifm.jl`: 28 checks on the same consistent sequence, over three slice shapes, all
  agreeing.

  Fixture `bifm_smoother`, the RxInferExamples model, small, without free energy. It agrees with v6
  in every posterior and rule call, with two things declared:
  - `:within_iteration`: v6 ran a slice's `znext` before its `in`, which read its cache;
  - `yt`'s posterior is requested, since only it reads the messages towards `out`, which v6
    computed whether or not anything asked.

  Independently of v6, an engine test builds the same model from Standard's `*`, `+` and normals,
  and BIFM's posteriors of `z` and `u` equal the RTS smoother's to 1e-8. A third test pins the
  free-energy error. `encode_fixture_value` learns `TerminalProdArgument`.

  The ported files have left `legacy/v6/`, with `rules/mv_normal_mean_precision/`.




**Exit criteria**
- [x] **delete, don't port** — `sphericalradial.jl`, `gausslaguerre.jl`, `importance.jl`,
      `laplace.jl` had no consumer; step 4 moved them and their tests to `legacy/v6/`, and
      they are not ported from there (skim the tests first, they may be the only record of
      intended behaviour)
- [x] the superseded `cvi.jl` (`ProdCVI`/`CVI`), `delta/layouts/cvi.jl` and
      `rules/delta/cvi/*` are not ported from `legacy/v6/` either. *Their dependencies are
      already gone:* step 4 moved `ReactiveMPOptimisersExt` to `legacy/v6/ext/` and dropped
      the `Optimisers` weakdep and `DiffResults` from `Project.toml`
- [x] delta node's built-in method set is now `{Unscented, Linearization}` — **an accepted
      capability regression**, alongside exported deletions; it needs (a) an explicit breaking
      entry in the release notes, not folded in with the renames, and (b) an error that
      names both the package to install and the method to switch to. First real customer
      for the diagnostics. Per #11, that is the host's `is_delta_node_compatible` guard
      carried over, with the error extended to name the alternative method
- [x] `CVIProjection` ships as a weakdep extension of the Delta node package *(step 2)*. Phase 0 found
      the layout collapse real but partial: dependencies absorb input selection, while
      static gating, the empty group and `q_out` aliasing are engine features
- [x] `MessagePassingRulesApproximations`: `Linearization`, the remaining piece. `Unscented`,
      `smoothRTS`, `approximations.jl` and `shared.jl` were ported in Phase 4.5 step 3, as pure
      numerics. **Standalone — must not depend on `MessagePassingRulesBase`**, nor on any
      distribution package. Utilities that algorithms use, not algorithms. The deps today are
      `LinearAlgebra` and `FastCholesky`; `Linearization` brings `ForwardDiff`, and the
      Gauss–Hermite cubature `FastGaussQuadrature` (entry brief). No `DiffResults` (it leaves
      with `cvi.jl`), no `Optim`
- [ ] numerical API carried over without broader redesign. FastCholesky is called directly
      until the numerical protocol (open item #13, parked) is settled
- [x] `ghcubature` and `approximate_meancov` move to `MessagePassingRulesApproximations` with
      `FastGaussQuadrature`, since Pólya, Probit and GCV all use them *(the Pólya package, until
      the entry brief counted their users)*
- [x] confirm `Optim` no longer appears anywhere *(step 1: only in prose and the pinned 6.5.0
      manifest)*
- [ ] non-standard nodes spun out, each into its node package (`INVENTORY.md`'s `node:X`): Flow,
      Autoregressive (with ConjugateAR), BIFM (with its `TerminalProdArgument` rules in
      `legacy/v6/src/rules/mv_normal_mean_precision/marginals.jl`), Pólya, ContinuousTransition,
      DiscreteTransition, GCV, Probit, SoftDot and GaussianCoupling (Delta already, in Phase 4.5
      case (d); the last four each get their own package rather than a shared `models` one,
      `DISCUSSION.md` §3.30). GP is not among them: it has no node in v6, and RxGP is its own
      package
- [ ] the helpers in `legacy/v6/src/helpers/algebra/` go with their users, as `INVENTORY.md`
      says: the permutation matrix to Flow, the standard basis vector and the companion matrix
      to Autoregressive, and `common.jl`'s helpers the Phase 6 nodes use to Standard;
      `legacy/v6/src/fixes.jl`, the `ForwardDiff` hot-fix, is dropped, since only the deleted
      `laplace.jl` and `cvi.jl` took a Hessian, so `legacy/` can go
- [ ] Pólya package carries the GPL-3 `PolyaGammaHybridSamplers`; ReactiveMP's MIT licence
      becomes honest again (see `PLAN.md` § Licensing)
- [ ] Probit and ContinuousTransition get their own algorithms from `ProbitMeta` and `CTMeta`
      *(Probit's, with its initial message and the dependencies page's section, in step 3)*,
      declaring their dependencies, and Probit's self-dependency a default initial message
      declared on its node (moved from Phase 5's `Require*` criterion); the dependencies page
      gains a section on initial messages
- [ ] surviving impure algorithms (BIFM and stateful projection algorithms) carry the
      `pure = false` marker under the agreed purity/RNG contract
- [ ] the packages stay in the monorepo under `lib/`; the split into repositories is Phase 8
      (`DISCUSSION.md` §3.40)

## Phase 7 — Complete the engine

Phase 4.5 built the engine's first cut and settled its design (restructured
2026-09-23, `DISCUSSION.md` §3.18); what remains here is completing it once Phases 5–6 have
ported the rules. The items that moved to Phase 4.5 are group stream wiring, the `Message`
envelope, edge order, `EdgeLabel.index` (#7) and the mixture `reverse` (#6).

Known scope:
- [ ] every node supported, as Phases 5–6 port their rules out of `legacy/v6/`. The mixtures'
      per-node `activate!` is already gone: step 4's generic activation replaces it
- [ ] `EqualityChain` `BitVector` → `Vector{Bool}`
- [ ] engine diagnostics: `check_everything_pure`, `check_everything_inplace`, checked buffers
- [ ] RxInfer adapted to the new engine, as its own major release: node and rule creation,
      a per-node `algorithm` option (replacing `meta` and v6's `where { dependencies = … }`),
      and default initial messages. What step 4 changed under it, all found by reading RxInfer
      5.5.2 (`src/model/plugins/`):
      - `factornode(fform, interfaces, factorisation)` takes `(name, variable)` /
        `((name, k), variable)` and a factorisation of names; RxInfer passes positions
        (`VariationalConstraintsFactorizationIndicesKey`) and would pass GraphPPL's
        `EdgeLabel.index` as `k` (#7);
      - `FactorNodeActivationOptions(; algorithm, postprocessor, annotations, callbacks)`
        replaces the six positional fields; `metadata`, `dependencies` and `rulefallback` are
        gone;
      - free energy: `score(T, FactorBoundFreeEnergy(), node, algorithm, pp)` takes the
        algorithm where it took `meta`, and `bethe_free_energy` can replace the assembly in
        `reactivemp_free_energy.jl`;
      - GraphPPL's node queries (`@node` traits, `sdtype`, `interfaces`, `alias_interface`,
        `nodesymbol_to_nodefform`) become `MessagePassingRulesBase.nodespec` and its accessors;
      - callbacks: a `MessageMapping` has `target` (`Target{:out}()`) and `algorithm` where it
        had `vtag`, `vconstraint` and `meta`. RxInfer itself does not read them, but user
        callbacks do (`compat/v6-comparison/record_engine_fixtures.jl` reads `vtag`);
      - the force-marginal plugin (`reactivemp_force_marginal_computation_plugin.jl:60-74`)
        calls v6's `marginalrule` with a `clustername` tag over `get_node_local_marginals`. It
        becomes a `MarginalMapping` with a `ClusterTarget`, as `activate_cluster!` builds a joint
        (`src/nodes/clusters.jl`), and the local marginals are now keyed `:μ`, `(:out, :μ)` or,
        for a whole group, `(:in,)`
- [ ] Aqua's `ambiguities` check re-measured on the new code and re-enabled, or its remaining
      pairs budgeted; it was 322 pairs on `main`, most in code now in `legacy/`
- [ ] the log-scale milestone, after the migration (user): v6's gaps were preserved
      deliberately, and this is where they are fixed, **or the feature is dropped**, decided
      then (`DISCUSSION.md` §3.37). Dropping it takes Mixture's rules with it, since its switch
      is a softmax over incoming log scales. Typed annotations (`Message{D, A}`, brief item 3)
      land with it, replacing the mutable `AnnotationDict`; kept, the log-scale key would get
      an owner there, in the base package
- [ ] the `.github/` workflows brought up to date before the first PR: they still describe
      the 1.10 matrix and the pre-step-4 layout (§3.22 left them alone), and `LibTests` has no
      job for `DeltaMessagePassingRules`
- [ ] explicit checks on scheduling order, annotations, retained values and free energy —
      not just numerical rule equality — for every ported node, against recorded v6 fixtures

---

## Phase C — Cleanup: historical remarks out of the repository

**Goal** (user, 2026-09-23): the released packages read as if written as they are. The
rewrite leaves remarks in code, docstrings, comments and tests that only make sense as its
history: phases and steps ("Phase 5, step 3", "case (d)"), the slice, what v6 did, pointers
into `DISCUSSION.md` or `PHASES.md`, `legacy/` paths. They are irrelevant once the rewrite is
done, and git keeps the history. Done last, after Phases 5–7 and before the release, so
that nothing written in between escapes it.

**Exit criteria**
- [ ] no mention of phases, steps, cases (a)–(d), the slice, `PLAN.md`, `PHASES.md`,
      `DISCUSSION.md`, `INVENTORY.md` or `legacy/` in `src/`, `lib/*/src`, `lib/*/test`,
      `test/`, `scripts/` or the docs; checked by a search that the cleanup commit records
- [ ] a remark that carries a reason keeps the reason, said in present terms: "the
      precisions come first because that is the update schedule", never "v6 did it this
      way" or "decided in step 3". A remark that is only history is deleted
- [ ] a comparison with v6 that still matters for users is kept where users read it, the
      release notes and the v6 → v7 guide (`docs/src/migration-guides/v6-to-v7.md`), not in
      code (ReactiveMP.jl#669 is such a case)
- [ ] the working documents go: `PLAN.md`, `PHASES.md`, `DISCUSSION.md` and `INVENTORY.md`
      (with `scripts/inventory.jl` and its `:quality` test item), and `CLAUDE.md` loses its
      § Ongoing work. What they decided that users need is already in the docs and
      v6 → v7 guide; the rest is in git
- [ ] `compat/v6-comparison` and its fixtures are removed, or kept as a named, documented
      migration aid, decided then; `legacy/` is already empty and deleted by Phases 5–6
- [ ] `CHANGELOG.md`'s `[Unreleased]` entries, which record the rewrite step by step, are
      replaced by release notes that describe the release

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
- [ ] the monorepo split into `ReactiveBayes/*` repositories, one per package, with their
      history, at registration (`DISCUSSION.md` §3.40; `PLAN.md` had it at Phase 6)
- [ ] package registration order decided, compat bounds set, supported Julia versions agreed.
      Work targets 1.13 only until then (§3.22); whether the 1.10 floor and its workarounds
      come back is decided here
- [ ] RxInfer's default package set updated
- [ ] documentation links across the three levels updated
- [ ] downstream migration readiness confirmed — the v6 → v7 guide exercised against a real
      external package (RxGP is the natural candidate)

---

## Open items

Tracked with stable numbers in `PLAN.md` § Open items. Of 14 items, #1, #3, #8, #9, #10,
#11, #12 and #14 are resolved (#3 and #9–#12 at the Phase 3 sign-off), and #4 is **deferred by
decision** (Phase 0; reopened only by a concrete ruleset use case). #13 is **parked** by the
user and holds back only the `linalg` context service. #6 and #7 were engine integration
requirements, closed in Phase 4.5 case (c) (#7's RxInfer side, passing `EdgeLabel.index`, is
Phase 7). #2 remains deferred unless needed. #5
(Reactant/StableCholesky) belongs to a separate effort and does not block this rewrite.

## Structural note

Rule kernels and test utilities can be developed independently of the engine. However,
**Phase 5 bulk migration is gated on Phase 4.5**, and release is gated on full engine and
downstream integration. Engine independence does not establish interface correctness.
Since the restructuring (`DISCUSSION.md` §3.18) the engine Phase 4.5 builds is the real one,
not a proof alongside v6, so Phase 5 ports rules straight into it and Phase 7 completes it.
