# Design discussion — rule/node system rewrite

Companion to `PLAN.md`. `PLAN.md` records *what was decided*; this records *why*, which
alternatives were considered and rejected, and which claims turned out to be wrong.

If you are an agent picking this up in a new session: read `PLAN.md` first for the
decisions, then the **Corrections** section below before proposing anything, because
several plausible-sounding ideas were already tried and shot down for concrete reasons.

Date of discussion: 2026-09-22. Branch: `refactor/rule-node-system-rewrite`.

---

## 1. The starting request

Three goals, stated as one:

1. Split the rule/node layer out of ReactiveMP so `AutoregressiveNode`, `RxGP` etc. can
   exist as independent packages — "make ReactiveMP more on-purpose instead of a can of
   everything".
2. Overhaul the rule/node system itself — it is old, archaic, carries unused features, and
   has weird syntax.
3. Support extra context in rules (cholesky strategy), preallocated output containers,
   more predictable dispatch, Reactant-implemented rules, and parallel-safe rules.

Scope was settled early: **clean break, new major version of everything, no backwards
compatibility, Julia floor may rise.** Package-split granularity (by family or otherwise)
was explicitly deferred. All the "bonus" goals were declared core, not nice-to-have.

---

## 2. What the investigation found

Numbers: 45 `@node`, 384 `@rule`, 106 `@marginalrule`, 57 `@average_energy`, ~24k SLOC,
490 rule definitions total, 50 rule directories, 172 `@call_rule`/`@test_rules` sites in
tests. (Corrected in Phase P: the earlier 48/390/108/52 counted the docstring examples in
`src/rule.jl` and `src/nodes/nodes.jl` as definitions.)

Findings that shaped the design:

- **`vconstraint` is a dead dispatch axis.** Hardcoded `Marginalisation()` at all five
  construction sites; `MomentMatching` is defined, exported, and dispatched on nowhere.
  447 occurrences of pure boilerplate.
- **The `m_`/`q_` prefixes are parsed, not just cosmetic.** The macro strips prefixes and
  recovers joint marginals by splitting `q_y_x` on `_`. That is *why* `@node` forbids
  underscores in interface names — an assert in the macro.
- **Introspection is string-scraped.** `showerror(::RuleMethodError)` decodes rules via
  `Base.arg_decl_parts` at fixed positional offsets (`decls[2..9]`). Any signature change
  silently breaks it.
- **Rules are not pure today.** `BIFMMeta` is mutated from inside message rules (`setH!`,
  `setΛz!`, `setμu!`, `setΣu!`). CVI stores ForwardDiff scratch in its meta.
- **Downstream already abuses `meta` as a mutable workspace.** RxGP's `UniSGPMeta` is a
  `mutable struct` with scratch matrices, a `counter`, and a `GPCache` of `Dict`s with
  `get!`-based lazy allocation. So "preallocation" is an existing folk pattern, done in the
  most thread-hostile way available.
- **Exactly one engine leak in the whole rules tree**: `rules/mixture/switch.jl` allocates a
  throwaway `randomvar` inside a rule to reach product-with-log-scale machinery.
- **The mixtures escape `@node` entirely.** `Mixture` hand-writes its node struct,
  `factornode`, `interfaceindex`, `activate!`, `interfaces`, `alias_interface`, `sdtype`,
  `collect_factorisation`. `GammaMixture` is a ~250-line clone of `NormalMixture`. Dead
  code found inside them: the three `…NodeFactorisation` singletons are never reached, and
  `Mixture`'s `RequireMarginal` path defines a 3-argument `functional_dependencies` that
  the 4-argument caller never finds — it would `MethodError` if anything reached it
  (~130 lines).
- **`RxGP.jl` already is the "external node package"** the split is meant to enable — it
  works today, but depends on all of ReactiveMP *and* RxInfer for what is pure math.
- Thread hazards, concretely: `EqualityChain`'s caches are a `BitVector` (bit-packed writes
  are read-modify-write on a shared word, so *neighbouring* indices race),
  `DeferredMessage.cache` can be double-forced, Rocket's `RecentSubject`/`AsapScheduler` is
  single-thread-assumed.

---

## 3. The design narrative, with pivots

### 3.1 Dispatch axes, and `algorithm` replacing `meta`

Started from: kill `vconstraint`, merge the four parallel name/type arguments into one
keyed container, add an explicit context.

**Pivot (user's idea, and the best structural move of the session):** `meta` is already two
things fused — an algorithm *selector* (`Unscented` vs `Linearization` vs `CVI`) and a
parameter bag (AR order, kernels, inducing points). Rename it `algorithm`, give it a
per-node default, make it swappable.

Why this was a big deal: it meant `Marginalisation`/`MomentMatching` did not need to be
deleted as a special case — that axis simply *is* the algorithm axis, so it disappears. It
later absorbed two more things (dependencies, purity), and turned
`RequireMessage`/`RequireMarginal`/`RequireEverythingFunctionalDependencies` from a
bolted-on mechanism into ordinary algorithms. *(In the end they are deleted, §3.21: their
dependencies go on a node's algorithm, and their initial values become initialization.)*

`context` was kept strictly separate and **non-dispatching** — it is infrastructure
(linalg strategy, RNG, buffers, services), not mathematics. An early draft had rules
dispatching on context; the user correctly rejected it. A later refinement kept from that:
rules *declare which context services they need*, which gives better diagnostics and
doubles as a manifest for a future traced path.

### 3.2 The rule syntax journey

This took the longest and went through three positions.

1. **Keep `m_`/`q_`.** Cheapest migration (near-`sed`), but permanently keeps the
   underscore ban on interface names, since joint marginals are recovered by splitting.
2. **`q(a, b)::T` with a generated binding name.** Proposed on the argument that mangling
   in the *generation* direction is safe while mangling in the *parsing* direction is the
   bug. **Killed by the user with a counterexample:** `q(a, b_c)` and `q(a_b, c)` both
   generate `q_a_b_c`. Generation is only unambiguous if names contain no underscores —
   which is the restriction the change was supposed to lift. The idea was circular.
3. **`m[x]` / `q[x]` — adopted**, and still the design; §3.14 later adds the colon
   (`m[:x]`) and moves the whole surface to keywords, but the container idea and its
   justification below are unchanged. The decisive property is that **declaration and body use
   the same spelling**. There is no derived name, so there is nothing to collide. The
   binding problem does not get solved, it disappears.

Consequences that fell out for free:
- `m[:μ]` and `q[:μ]` in one signature is unremarkable — two separate containers. An earlier
  design (from a subagent) had invented a `Both{...}` wrapper type and a
  "key a cluster by its lexicographically smallest member" rule to handle this in one
  merged container; all of that evaporated.
- `q[:y, :x]` is a structural cluster (lowers to `getindex(q, Val((:y,:x)))`), so nothing is
  ever split on `_` and interface names may contain underscores again.
- `m[:inputs...]` expresses a variadic group, replacing `ManyOf{N,T}` and its `where {N}`.

Also considered and rejected: a `where { q(a,b) <: T, m(x) <: S }` block (user's proposal).
It parses fine — verified with `Meta.parse`, including mixed type-parameters and slots in
one brace list — and `<:` is more *honest* than `::` (the macro already converts `::T` into
`Message{<:T}`). Rejected because `where` already means type parameters to every Julia
reader, and rules genuinely need the original meaning (delta rules carry
`where {N, M <: Unscented, L, I <: NTuple{L, Function}}`).

*(Superseded by §3.14, which adopts the full-keyword form for rules too — `args = (...)`
with the body as an ordinary lambda. The paragraph below records the position that was held
first; the reason it was reversed is in §3.14, not here.)*

Keyword options (`algorithm`, `pure`, `inplace`) go in the header; the signature stays
positional. Full-keyword form (`arguments = (...)`) was considered and rejected for `@rule`
— it buries the payload — but **adopted for `@node`**, which has far more to grow into
(aliases are already kwargs-in-disguise there).

### 3.3 Dependencies as a language

Started as: variadic groups need a way to say which members a rule receives. Four modes
were observed in-tree — whole group (`Mixture(:out)`), leave-one-out (`DeltaFn`), aligned
element (`NormalMixture((:m,k))` needs `p[k]` only), and none (`Mixture((:inputs,k))`).

First insight: those are not four policies, they are one index selector parameterised by
the target's index.

**Pivot (user):** make it a *language*, and make it a property of the **algorithm**, not
the node — a node declares a default algorithm; users can write custom algorithms with
custom dependencies. Written in the same `m[]`/`q[]` vocabulary as rules, which means the
dependency spec and the rule signature are two statements of the same fact and can be
**cross-checked at load time**. That closes a failure class that currently surfaces as a
confusing runtime `RuleMethodError` with no hint that the *dependency spec* is what's wrong.

**Initial pivot (user):** custom dependencies could imply factorisation rather than derive
from it. The original condition required all requested marginals to form a partition.
**Review correction:** auxiliary beliefs need not be entropy clusters, so that condition
conflated inputs with scoring. The precise relationship, ordering and conflict handling
remained open in `PLAN.md` item #9 until the Phase 3 sign-off resolved it (§3.16).

**Constraint discovered:** selectors must have **statically known output arity**, otherwise
the `ManyOf` length becomes runtime-dependent and `ManyOf{N,T}` cannot specialise. The
user's `k -> (k-1,)` chain example violates this at `k=1` (arity 0 or 1). Resolutions:
total selectors (wrap/clamp/pad), or a distinct target type for the boundary.

### 3.4 In-place rules — two wrong turns, both mine

This is the part where the assistant was wrong twice and the user corrected both times.

**Wrong turn 1: "observe the shape, don't compute it."** Proposal was to run the allocating
rule once on sweep 1 and keep the result as the buffer. **Circular** — if only the in-place
rule exists, there is nothing to run.

**Wrong turn 2: derive the in-place form by rewriting the body's tail into
`copyto!(out.μ, mexpr)`.** This defeats the entire purpose: `mexpr` has already been
allocated by the time you copy it. In-place exists precisely so the intermediate never
exists.

**Landing point (user's `@allocate` idea):** *(the reasoning below stands; the spelling is
superseded — `@allocate` became the `preallocate` keyword in §3.14)* the buffer shape must
be *declared*, because
it is derivable from the input types and nothing else can supply it. This is exactly the
`rand`/`rand!` relationship in Distributions.jl — `rand` can delegate to `_rand!` only
because it knows how to build the container. `@allocate` is that knowledge.

**Wrong turn 3 (also mine, caught by the user): "`rule!` allocates nothing in steady
state".** In-place ≠ non-allocating. In-place means the *result* is written into a provided
container; intermediates may still allocate. Non-allocating is a stricter, separate,
opt-in property.

Refinement on top of `@allocate`: the allocation primitives should **dispatch on the source
type** (`buffer_like(x)`), not allocate a default and post-process. Post-processing an
`SVector` case into an `MVector` would repeat the `copyto!` mistake one level up.
`buffer_like` then doubles as the device seam (see 3.7).

User's semantic for naming the buffer *(superseded by §3.14: `output` is now its own body
slot, so the restriction is lifted)*: in an `inplace` rule, `m[<target edge>]` **is** the
output buffer, and such a rule may not also require the inbound message on that edge. This
resolves an ambiguity the assistant had flagged as a bug, and has a bonus: it pins the
output type in the signature, making `@allocate` type-checkable.

### 3.5 Purity — from "testable" to "declared and audited"

Assistant initially proposed testing purity by running rules twice and hashing the
algorithm object. **User rejected:** running twice tests determinism, not purity, and
hashing is unreliable for mutable structs holding arrays. Purity is the rule author's
responsibility.

Assistant conceded — then over-corrected to "purity is not testable, full stop".

**User's synthesis, which is the right answer:** you cannot *prove* purity, but you can
**audit the declarations at run time**. Hence `check_everything_pure = true` on the
inference engine, erroring with the name of the offending rule. The motivating case was
investigating side effects while differentiating through inference. **Review correction:**
purity is not a general prerequisite for ForwardDiff, and passing the audit does not prove
gradient correctness. Requiring it in a checked workflow is a project policy; derivative
tests are separate. The audit sits alongside `check_everything_inplace` and checked buffers.

Inheritance model (user's): declared on the algorithm, inherited by its rules, overridable
per rule. `BP` is pure *(now `DefaultAlgorithm`, §3.20)* so ~350 rules need no annotation; BIFM and, in the original design,
old CVI mark the algorithm once rather than each rule. CVI is subsequently removed (§3.9b-ii).
Assistant added: narrowing is safe, **widening must taint the graph-level check** or the
inheritance is unsound in the direction that matters.

### 3.6 Threading — descoped, correctly

Assistant's first pass (via a subagent) proposed an explicit schedule IR with layered
fork-join. **User pushed back:** they are not asking for a scheduler rewrite; they already
did parallel `materialize!` with a student and it worked. The requirement is simply that
**rules be pure**.

Accepted. What survives from the larger analysis is only the concrete hazard list (the
`BitVector`, `DeferredMessage.cache`, mutated metas) and the observation that
`check_everything_pure` is the natural safety interlock before threading is switched on.

### 3.7 Reactant — descoped, but after a correction

Assistant (via subagent) called per-rule Reactant compilation "a dead end" — a Gaussian
rule is 2–3 BLAS calls versus seconds of compile time per shape.

**User explicitly disagreed:** it is research, build it so it can be explored.

Assistant conceded the framing was wrong: the analysis supported "XLA won't beat OpenBLAS
on a 5×5 Gaussian rule on CPU today", which is a benchmark claim about one regime, not an
architectural verdict.

The technically useful outcome: **in-place discipline and traceability discipline are
nearly the same discipline** — both forbid materialising intermediates, both need static
shapes, both need the output container declared up front. So `preallocate` +
destination-passing is already the right shape, and `buffer_like` dispatching on array type
*is* the device seam. Reactant is not a second path; it is the same path with a different
array type.

Final scope (user): Reactant and StableCholesky are **separate efforts, not in this plan**.
Keep `buffer_like` extensible and documented; build nothing else.

### 3.8 The mixture investigation

Triggered by the user calling the mixture implementation "absolutely horrible". It is, and
the reason is diagnostic: **one missing concept — the variadic interface group.**

Verdict from a full read: ~70–85% of the three mixture files is boilerplate a group concept
would delete. `Mixture`'s custom `activate!` *is* the generic loop, differing only by
skipping `initialize_clusters!`. Its `collect_latest_messages` is a hand-rolled, less
efficient re-implementation of `combineLatestUpdates` + `combineLatestMessagesInUpdates`,
primitives that already exist and that delta already uses.

Three concrete blockers in the generic machinery, all small: the adjacent-duplicate check
errors on a group's wire format, the arity check demands fixed arity, and
`__collect_latest_updates` flattens names so `N=2` produces `Val{(:out,:switch,:m,:m,:p,:p)}()`.

Irreducibly custom afterwards: group *alignment* (`m_k` pairs with `p_k`), construction
validation, and `ManyOf`-aware entropy.

Two traps recorded: the mixtures' `getinterfaces` **reorders** edges, but GraphPPL's
factorisation is positions into the original flat list — a generic implementation must
preserve order or translate. And there is an unexplained `reverse(...)` plus a means/precs
swap in the marginal wiring that should be pinned by a regression test before anyone
touches it.

Good news: GraphPPL already supports multiple, non-trailing variadic groups, so
`[out, a..., b..., x]` works end to end.

### 3.9 Packages

Current package names: `MessagePassingRulesBase` + `StandardMessagePassingRules` +
`MessagePassingRulesApproximations` + `MessagePassingRulesTestUtils` +
`DeltaMessagePassingRules` (since Phase 4.5 case (d), §3.25) + the `ReactiveMP` engine. The approximation package's contents and layering changed below.

**Pivot (user):** test tooling is its own **package**, not part of the base. The assistant
had proposed a `Test` package extension; the user pointed out testing deps belong in
`[extras]`. The assistant's `weakdeps` remark was explaining why its *own* extension idea
was bad (an extension's deps must be weakdeps of its host, so cubature/Turing would land in
the base's `Project.toml`) but stated it confusingly. Outcome: separate package, listed
under `[extras]` by consumers, no weakdeps or extensions involved anywhere. *(On 1.10 an
unregistered extra could not resolve, so Phase 4 and 4.5 wired it through a test environment;
on 1.13, `[extras]` plus `[sources]` does it, §3.22.)*

**Superseded rationale:** the assistant initially treated approximation methods as
algorithms and proposed moving seven numeric dependencies with them. The later decision
separates numerical utilities from algorithms and deletes unused methods (§3.9b).

**Hard constraint (user, stated twice, firmly): `MessagePassingRulesBase` must not depend on
`ExponentialFamily`. BayesBase exists precisely for this.** If something is missing from
BayesBase, add it to BayesBase.

### 3.9b Approximations — a package that changed shape twice

The assistant proposed `MessagePassingApproximations` off the back of the user's unease
about depending on cubature packages ("old… huge… only create extra dependencies"). Since
`algorithm` makes approximation methods first-class values, the inference was that they
could be lifted out wholesale, taking seven heavy numeric deps with them. **This was never
explicitly agreed and was presented in the package table as if settled — an overstatement.**

Checking actual usage changed the picture twice.

First: `Unscented` and `Linearization` are used only by the delta and flow nodes, `CVI` only
by delta, and `GaussHermite`, `SphericalRadial`, `GaussLaguerre`, `Laplace`,
`ImportanceSampling`, `srcubature`, `glcubature` have **no consumer in `src/` outside
`src/approximations/` itself**. This establishes no internal consumers, not no downstream
users: several are exported API. The agreed deletions remove dependencies, but need explicit
migration entries, including cases with no replacement (`PLAN.md` item #14, since resolved
in Phase P — `INVENTORY.md` carries a migration note for all 18 exported deletions).

Second, a correction: the assistant reported `rts_smoother` as unused. **That name does not
exist** — the function is `smoothRTS`, and it is load-bearing, used by
`rules/delta/unscented/marginals.jl` and `rules/delta/linearization/marginals.jl`. A bad
grep, not a real finding.

**Decision at this stage (CVI subsequently removed in §3.9b-ii):** keep `Unscented`,
`Linearization`, `CVI` (plus `smoothRTS`) in a
`MessagePassingRulesApproximations` package that delta and flow depend on; delete the rest
outright; `ghcubature` follows `multinomial_polya` into the Pólya node package.

**User's decision on layering, correcting the assistant:** these are **not** algorithms and
the package **must not** depend on `MessagePassingRulesBase`. They are a utility API that
algorithms use under the hood. The assistant had argued the opposite (make them algorithms,
depend on base) on the grounds that they are already shaped like the algorithm axis. The
user's framing is better: "how do I approximate this integral" is numerics; "which rules
run" is an algorithm *(originally "which update scheme am I running"; algorithms are not
schemes, §3.20)*. The delta algorithm *uses* Unscented; it is not
Unscented. Keeping them siblings leaves the numerics usable outside this ecosystem.

API: no broader numerical redesign. Replace global `cholinv` calls through a minimal
numerical protocol without depending on the base package. Its representation remains open
(`PLAN.md` item #13); the earlier wording simply said to thread `ctx` through.

### 3.9b-ii CVI is superseded; the delta-layout hypothesis

Follow-up to the approximations discussion. The old CVI method (`ProdCVI`, aliased `CVI`,
~2022) is **superseded by `CVIProjection`** (~2024) and gets removed. This is not a
judgement call — `ProdCVI`'s own docstring already carries the note *"`ProdCVI` is
deprecated in favor of `CVIProjection`"*.

Worth recording because it is counter-intuitive: `cvi_setup!` / `cvi_update!` and the whole
`ReactiveMPOptimisersExt` belong to the **old** method, not the new one. That extension
predates `CVIProjection` by more than a year and exists solely to supply those two hooks, so
it is deleted along with the `Optimisers` weakdep and `DiffResults` (`cvi.jl` is its only
user). Anyone reasoning from those names will guess the wrong way round.

One user-visible consequence is **an accepted capability regression**, alongside the
exported approximation deletions. The delta node's out-of-the-box approximation set
shrinks from `{Unscented, Linearization, ProdCVI}`
to `{Unscented, Linearization}`, because `CVIProjection` requires
`ExponentialFamilyProjection` to be loaded. So a model that runs today on a plain
`add ReactiveMP` may afterwards need an extra install rather than an edit.

Options weighed: accept and document; make RxInfer depend on `ExponentialFamilyProjection`
so the capability never disappears at the level where users actually live; keep something
sampling-based in the default install; or treat it purely as a diagnostics problem.

**Decision: accept and document, plus fix the diagnostics** — an explicit breaking entry in
the release notes rather than one line among the renames, and an error that names both the
package to install and the method to switch to. This tests the diagnostics, but loaded-rule
discovery alone cannot identify an unloaded extension. Capability metadata must be available
in the already-loaded host or a static table (`PLAN.md` item #11). *(Settled at the Phase 3
sign-off, §3.16: the host's existing `is_delta_node_compatible` guard already is that
metadata; no table.)*

**The layout hypothesis (open, moderate confidence).** `CVIProjection` currently spans the
type (in `approximations/`), rules (in the extension) and a *layout* (also in the extension,
but engine code — it builds `MessageMapping`, calls `connect!`, wires Rocket streams). That
straddles two future packages, which is what made its placement awkward.

The observation that may dissolve the problem: `AbstractDeltaNodeDependenciesLayout` looks
like a bespoke version of the dependency language. **The original count missed the
known-inverse variant:** there are four layouts (default, known-inverse, CVI, CVI-projection),
implementing or delegating the same four targets. The CVI-projection docstring reads like a
dependency spec, but stream aliasing, static-input gating and initialization must also be
preserved. The earlier claim that everything else was wiring boilerplate is unproven.

**Conditional path: if Phase 0 validates the collapse, replace layouts with dependency
declarations (C), then ship `CVIProjection` as a weakdep extension of Delta (A).** An earlier draft
argued for a standalone package; that was over-engineering for a leaf component of a few
hundred lines. The blocker for the extension was never fundamental — it was only that the
layout half is engine code with nowhere to live in a node-package extension. Removing the
engine half removes the blocker. A standalone package remains a cheap later upgrade if
anyone needs to depend on the rules, if compat coupling forces awkward releases, or if it
grows its own CI; promoting an extension to a package is much easier than the reverse.

**Test the hypothesis in Phase 0**, not at Phase 6. The delta layouts are the hardest thing
the dependency language would have to express, and finding its limits on the hard case is
exactly what the spike is for.

### 3.9c Licensing — found, accepted, deferred

The user recalled a licensing problem and it checked out, though more directly than
remembered: it is not one of Pólya's dependencies, it is **`PolyaGammaHybridSamplers`
itself, which is GPL-3**, entered as a plain `[deps]` entry while ReactiveMP ships MIT. Two
nodes use it (`multinomial_polya.jl`, `rules/binomial_polya/beta.jl`), and it propagates
downstream to RxInfer.

**User's decision: live with it.** The discrepancy has existed for over a year; this rewrite
resolves it rather than a separate fix, because the Pólya nodes move to their own package
which may be GPL-3, leaving ReactiveMP honestly MIT.

### 3.10 Naming

**User's decision:** rename everything, since the redesign is total anyway and generic
names invite collisions when downstream packages add methods.

- Generic functions, not exported: `message_passing_rule`, `message_passing_marginalrule`,
  `message_passing_average_energy`, `!` variants for in-place.
- Definition macros, exported, deliberately long (typed once per definition):
  `@define_message_update_rule`, `@define_marginal_update_rule`, `@define_factor_node`,
  `@define_average_energy`.
- Assistant's addition: **in-body macros need not be exported at all.** *(Superseded by
  §3.14, which deletes the category entirely rather than choosing how to export it.)*
  `@allocate` and
  `@logscale` only appear inside a rule body, so the enclosing definition macro can
  recognise and rewrite them. Short names, zero namespace footprint, and using one outside
  a rule body becomes a clean error.
- Invocation macros stay **short** — the "long but unambiguous" argument inverts for
  something typed constantly at a REPL in front of students.

### 3.11 Testing and documentation

Strict TDD, failing test first in PRs unless justified.

The leverage point: **because the definition macros emit data, "every rule has a test" becomes a
CI check** rather than a review norm. Nothing in the current system can do this, since rules
exist only as methods and enumerating them means string-parsing `methods()`.

Tooling: **stay on TestItemRunner** and teach `runtests.jl` to filter by name and tags — it
already receives them (this supersedes an earlier ReTestItems recommendation, see
Corrections 7d); Runic (deterministic, kills the formatter-drift problem the current
`Makefile` documents); Aqua checks re-enabled; JET expanded well beyond its current two
uses.

**Node-definition verification** — the idea the assistant rates highest and that was
prompted by the user asking whether sampling could verify rules generically. The existing
tables are **golden-value tests**: they lock in whatever a rule produced the day it was
written, so a rule wrong from birth has a passing test forever. Computing the reference
update numerically from `nodefunction` (BP as the integral, VMP as `exp(E[log f])`) tests
the *mathematics*. The machinery already exists in `src/approximations/`.

The user's Turing variant carries a subtlety the assistant flagged: a BP message is not a
posterior, so sampling the local factor gives a *marginal*. The comparison must be
`message × a known proper test prior` against the MCMC marginal under that same prior,
otherwise it silently measures the wrong thing and appears to work on symmetric conjugate
cases.

`check_type_promotion` becomes default-on. The user's reason is better than the assistant's
framing: it is not a float-width check, it is a container- and number-type propagation
check, and it is what makes `ForwardDiff.Dual` work through a rule.

Documentation: docstrings + jldoctests + Documenter `@example` blocks, all run in CI.
Everything with a docstring must appear in the docs (enforced by `checkdocs = :all`);
genuinely internal helpers use comments instead. Layered across three packages — reference
in the base, engine mechanics in ReactiveMP, conceptual/tutorial in RxInfer.

### 3.12 Educational and interactive use

Raised late by the user: the system is taught in the BMLIP course at TU/e, and invoking
rules manually is a good teaching device. Promoted to a first-class goal.

Almost free, because it is all registry queries: rule listings, `@which_rule`, a
**coverage matrix** (edges × algorithms — rule sets, not schemes, §3.20 — showing which rules exist) that serves students
and developers equally, `Base.show` MIME methods for REPL and notebooks, and visualisations
behind the extension mechanism following GraphPPL's pattern. Error messages count as
pedagogy — the near-miss display with per-slot diffs is a teaching tool.

---

### 3.13 Phase P decisions

Made while executing the preparation phase, all grounded in measurements taken at the time.

#### Benchmarks deleted rather than repaired

The `benchmark/` suite was removed. It covered exactly one node, its `ContinuousTransition`
half was never included by `benchmark/rules/rules.jl` (and called `StableRNGs(42)` — the
module, not the constructor — so it would have errored if wired in), `scripts/bench.jl`
called `BenchmarkTools.judge(::Module, ::String)` which is a **PkgBenchmark** method, no CI
workflow ran it, and its output paths were gitignored so no result was ever retained.

The user's call, and the right one: this is not a baseline, it is the appearance of one.
Performance verification moves to **RxInferBenchmarks.jl** at implementation time, where
there is a new engine to measure against. Phase P therefore captures no performance
baseline, which is an honest gap rather than a hidden one.

#### Standard versus models

`PLAN.md` defined `StandardMessagePassingRules` only as "standard distribution nodes +
arithmetic", which left about a dozen nodes unclassified. **User's decision:** standard
holds what is generic and model-agnostic — distributions, arithmetic, logic, mixtures —
and domain-specific models (`GCV`, `Probit`, `SoftDot`, `GaussianCoupling`) go to a sibling
package. `ContinuousTransition` gets its own package rather than joining them.

The package's **name is deliberately deferred to Phase 6**, when it is built. The inventory
records the placeholder token `models` and says so explicitly, rather than inventing a name
to look finished. *(Superseded by §3.30: each of the four gets its own node package.)*

#### The Julia floor stays at 1.10

The assistant flagged that `PLAN.md`'s context design said "a `ScopedValue` supplies the
default at the engine boundary", that `Base.ScopedValues` is 1.11+, and therefore that the
floor had to rise. **The user rejected the premise:** there is no reason to use
`ScopedValues` for that at all — the context is an ordinary object constructed once and
passed into the rules, most likely held by `MessageMapping`.

That is correct, and the assistant's framing was wrong in a specific way worth recording:
it treated one sentence of a draft as a constraint on the whole project. A plain default
argument gives the same behaviour on 1.10. The floor should move when something concrete
needs it to, and nothing does. `PLAN.md` § Dispatch axes now says this directly.

*(Superseded in Phase 4.5, §3.22: work targets Julia 1.13 only, for the tooling, not for
`ScopedValues`; the floor is reconsidered at registration.)*

#### Ambiguities are five problems, not one

322 ambiguous pairs sounds like a redesign-scale problem. It is not: **253 of them come
from three files in `src/helpers/algebra/`** — custom array types declaring `*` and `dot`
against bare `AbstractMatrix`/`AbstractVector`, colliding with ArrayLayouts, PDMats,
FillArrays and LinearAlgebra. That is engine-side helper code with nothing to do with the
rewrite.

Only **27** involve rule dispatch, and they are all one repeated shape: the delta catch-all
`rule(::F<:Function, …, meta::DeltaMeta, …, node::DeltaFnNode)` against the arithmetic-node
rules, where neither method is more specific. That is precisely the class the new dispatch
design claims to eliminate, which turns a vague aspiration into a Phase 0 target with a
number attached.

#### Two findings from building the inventory

**`CompanionMatrix` is dead.** No reference in `src/` *or* `test/`. The user's instinct was
that the autoregressive node must use it — reasonable, since AR genuinely does use a
companion-matrix representation, but through its own `ARTransitionMatrix`
(`autoregressive.jl:270`), which superseded the shared type and left it exported and
untested. It accounts for 75 of the 322 ambiguities, so deleting it is worth more than its
line count suggests. *(Wrong, found by the Phase 6 entry brief: AR builds it with `as_companion_matrix` in its `x`, `y`, `γ` and marginal rules; the search looked for the type's name, not its constructor. It goes to the AR package.)*

**One recorded claim was checked and survived.** §5 says `Optim` leaves ReactiveMP entirely
because only `laplace.jl` uses it. A first grep appeared to contradict this by matching
`continuous_transition.jl` and `approximations/optimizers.jl` — but those were the words
"Optimized" and "Optimizer", not the package. The claim holds. Recorded because the
near-miss is the kind of thing that gets published as a correction when it is simply a bad
grep.

---

### 3.14 The macro surface becomes keyword-based

Prompted by a small observation with a large consequence: why does
`NormalMeanVariance(:out)` carry a colon when `m[mu]` does not? The inconsistency was real.
The user's first instinct was to drop the colon from the outbound edge; pulling the thread
produced a different and better answer.

#### The design

Everything becomes a keyword, and the body becomes **an ordinary Julia lambda over a real
arguments object** rather than a body the macro rewrites:

```julia
@define_message_update_rule(
    node    = NormalMeanVariance,
    target = :out,
    args    = (m[:μ]::PointMass, m[:v]::PointMass),
    body    = (args) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v])),
)
```

#### Why this forces symbols rather than permitting them

This is the part worth remembering, because the conclusion is the opposite of the intuition
that started it. In a real lambda, `args.m[μ]` is an `UndefVarError` — `μ` is not a
variable. Only `args.m[:μ]` works. So:

1. the body is *forced* to symbols;
2. the declaration must match, because declaration/body agreement is the entire reason
   `m[]`/`q[]` was adopted over name mangling (§3.2);
3. consistency then carries symbols into `target = :out` and into
   `@define_factor_node(interfaces = [:out, ...])`.

Under the old macro-rewritten body, bare names were fine and the colon really was
decoration — which is exactly why it looked inconsistent. Choosing a real lambda body makes
the colon load-bearing. The right observation, the opposite conclusion.

#### Parsing facts, checked rather than assumed

- `m[:inputs...]` **parses** (`head = ref`). The worry that it would not was unfounded.
- `q[:p[:k]]` parses, but as `(:p)[:k]` — *indexing a Symbol*. So a group member is spelled
  **`q[:p][k]`**, which parses as `(q[:p])[k]` and matches runtime access exactly. This is
  better than the previous `q[p[k]]` regardless of the colon question.
- An indexed target is `target = (:m, k)`, matching today's `@rule NormalMixture((:m, k))`.
- Keyword macro arguments arrive as `Expr(:(=), name, value)`, order-independent.
- Body parameter names extract reliably from the lambda, including typed ones.

#### What it deletes

Both "in-body macros" disappear rather than being renamed. They were never really macros —
they were tokens the enclosing macro rewrote, which is why using one outside a rule body
failed confusingly. `@allocate` becomes the `preallocate` keyword; `@logscale` becomes
`annotate!(ann, :logscale, v)`.

**The in-place/inbound-message restriction is also dropped** (user's observation). It existed
only because `m[target]` had to mean either the output buffer or the inbound message and
could not mean both. With `output` as its own body slot, `output` and `args.m[:out]` are
distinct bindings and a rule may use both.

The old restriction had a stated benefit — it pinned the output type in the signature. That
benefit is preserved by a better mechanism: the ordinary typed lambda parameter
`(output::MvNormalMeanPrecision, args) -> ...`, which **Julia itself** enforces
(`MethodError` on mismatch) and infers, rather than macro-side analysis.

#### The `RuleSpec` is the execution vehicle, and carries no type parameters

The user's design: dispatch resolves to a `RuleSpec` which **stores the body and the
`preallocate` lambda** and knows how to call itself, so the engine never branches on
`inplace`. The same object the registry holds for `@which_rule`, the coverage matrix and
`check_rules()` is the one that runs, so the two cannot drift. It can also carry the body's
**source text**, which is what lets `@which_rule` show a rule rather than merely name it.

The assistant initially warned that storing the body in the spec would make every invocation
a dynamic call. That warning was too strong — it is true only where the compiler cannot see
which body is in the field, and the first measurements that seemed to prove it were
contaminated by constant folding.

**The user's decision: no type parameters at all.** `RuleSpec` is a plain immutable struct —
`body::Function`, `prealloc::Function`, `inplace::Bool`, `pure::Bool`, source text, file,
line, plus the registry metadata. *"It's much easier for Julia to do something like
`find_rule(args)` that returns a type-stable `RuleSpec` and then use it, instead of
`find_rule(args)` returning an abstract type."*

That is the right axis to optimise. The parameterised alternative makes every rule a distinct
`RuleSpec{B, P}`, so a lookup that cannot statically pin down which rule fires returns a
*union* of spec types — the thing that is not type-stable is the parameterised version, not
the plain one. Booleans hoisted into signatures are a known downstream-instability trap, and
the same objection applies to hoisting a closure's type.

**Measured, for the record** — on 1.13, with resolution held inferable but *not*
constant-foldable, at a call site that can reach two different rules:

| representation | `find_rule` returns | inferred | allocations |
|---|---|---|---|
| no type parameters, body called through the field | `RuleSpec` — concrete | `Any` | 48 |
| `RuleSpec{B, P}`, body called through the field | `Union{RuleSpec{…}, RuleSpec{…}}` | `Float64` | 32 |
| no type parameters, spec as data + generated execution method | `RuleSpec` — concrete | `Float64` | 0 |

Two separate findings fell out of getting this measurement right:

- **`inplace` as a type parameter buys nothing.** Measured identical to a plain `Bool` field —
  zero allocations either way, including with the split body shapes this design uses, where
  an allocating body takes `(args)` and an in-place body takes `(output, args)`. The arm that
  does not apply typechecks harmlessly. It also adds no distinct spec types, since `B` is
  already unique per rule and the flag is functionally determined by it. Pure cost.
- **The published reproduction below was one-sided and would mislead.** Run as written, the
  good case reports 0 — but so does an obvious `::Function`-field counterpart, because a spec
  constructed inline inside an inlinable `resolve` is constant-folded away entirely. Any
  comparison must defeat constant folding (`@noinline` resolution returning a
  runtime-selected spec) and must report a call site that can reach more than one rule, or it
  measures the best case and calls it the design.

**Decision: the indirect call is accepted and revisited later**, with real rules rather than a
toy. The two alternatives above stay on the table, and neither changes the macro surface,
which is what would actually be expensive to move. Phase 0 reports the number; a number is
something to act on, pre-optimising the struct is not.

**What holds regardless:** resolution must not go through a runtime container. A spec fetched
from a `Dict` keyed on runtime values infers as `Any` whatever the spec's own type — so the
per-module `const` plus dispatch in § Registry is a requirement, not a preference.

Reproduce (adapt `find_rule` per row of the table; as written this is the parameter-free
representation the design adopts):

```julia
struct RuleSpec
    body::Function
    prealloc::Function
    source::String          # introspection payload
    pure::Bool
    inplace::Bool
end

const body1 = (args) -> args.a + args.b
const body2 = (args) -> args.a * args.b
const pre1  = (args) -> zeros(2)

# `@noinline` + a runtime-selected spec: defeats constant folding, so the figure below is
# the one that can actually appear in an engine. An `@inline` resolve returning one literal
# spec folds the whole thing away and reports 0 for every representation.
@noinline find_rule(::Val{:N}, ::Val{:out}, flag::Bool) =
    flag ? RuleSpec(body1, pre1, "(args) -> args.a + args.b", true, false)
         : RuleSpec(body2, pre1, "(args) -> args.a * args.b", true, false)

function call_rule(node, target, args, flag)
    spec = find_rule(node, target, flag)
    return spec.inplace ? spec.body(spec.prealloc(args), args) : spec.body(args)
end

function bench(flag)                   # measure inside a function: a non-const global
    a = (a = 1.0, b = 2.0)             # would box and report a spurious 16 bytes
    call_rule(Val(:N), Val(:out), a, flag)
    return @allocated call_rule(Val(:N), Val(:out), a, flag)
end

bench(time() > 0)                      # 48 -- and 0 if the call site can reach only one rule
code_typed(call_rule, (Val{:N}, Val{:out}, NamedTuple{(:a, :b), Tuple{Float64, Float64}}, Bool))
```

#### Accepted costs, stated plainly

- **Verbosity.** One line becomes roughly six, across ~490 rule definitions — nearer ~590
  once the 57 `@average_energy` and 45 node definitions take the same surface. Chosen deliberately:
  one uniform surface, no positional shorthand, no second grammar to document or migrate.
- **The `m` collision.** Containers stay `m` and `q`, and `NormalMixture` has an interface
  group named `m`, so `args.q[:m][k]` uses `m` in two senses. Accepted in exchange for one
  short vocabulary.
- **The `args = (...)` declaration is still a mini-language** the macro parses. It is a
  signature, so this is unavoidable. "No magic" holds in the *body*, not everywhere. Saying
  otherwise would overstate what changed.

---

### 3.15 Phase 0 — what the spike found

The spike lived in `spike/` and was deleted when Phase 0 closed. The whole tree is present
at `81822c57` (`git show 81822c57:spike/README.md`). Everything worth keeping is here.

#### The representation gate

Measured on the **1.10 floor** and on 1.13, with the adopted parameter-free `RuleSpec`:

| | 1.10.12 | 1.13.0 |
|---|---|---|
| `args.m[:sym]`, `args.q[:p][k]` | 0, inferred | 0, inferred |
| routing, call site reaching one rule | **0**, `Float64` | **0**, `Float64` |
| routing, call site reaching two rules | 48, `Any` | 0, `Any` |
| same, parameterised spec, two rules | 32, `Float64` | 0, `Float64` |
| `NormalMeanVariance(:out)` end to end | **0** | **0** |
| in-place kernel with a provided buffer | **0** | **0** |
| cold compile, one rule | 5.8 ms | — |
| warm execution | 1.33 ns | — |

So the indirect call the parameter-free decision accepts **costs nothing on the ordinary
path**: a factor node's functional form is fixed, so its call sites reach exactly one rule.
It costs 48 bytes on the floor only where resolution is genuinely ambiguous, which is where
type information has already been lost upstream. `find_rule` returns a concrete `RuleSpec`
in every case, which was the point of dropping the parameters. JET reports nothing on the
routing. `Float64`/`Float32`/`BigFloat` all propagate without widening or allocation — the
property that makes `ForwardDiff.Dual` work through a rule.

Specialization growth is **multiplicative in (group size × element type)**: six group sizes
added 20 specializations. That is a property of the tuple rather than of this design —
`ManyOf{N,T}` has it today — and it is the number to watch if compile time becomes the
complaint during Phase 5.

#### Measuring this is easy to get wrong — five ways, all hit in practice

Recorded because every one of them silently changes the answer, and two of them were caught
only because a gate failed:

| mistake | effect |
|---|---|
| measuring against a non-`const` global | `+16` bytes of boxing |
| a varargs helper that splats, `f(xs...)` | `+48` bytes of its own, in every figure |
| constructing the spec inline in an inlinable `find_rule` | constant-folded away — `0` for *every* representation |
| closing over the node in a loop, so it is a `DataType` rather than `Type{Node}` | the call goes dynamic and reports `Any` |
| defining the rule before the timing function | the caller's own compilation absorbs it; cold time reads as zero |

A gate that asserts only the flattering direction is vacuous. The rule taken: report both
call-site shapes, and include a negative control that must allocate.

#### Three findings that change the design

**1. The body slots need the target threaded through.** `PLAN.md` listed six slots
`(output, algo, ctx, args, ann, node)` *(five since §3.16)* and no target — but an indexed target
`target = (:m, k)` has to bind `k`, which is a runtime value the lowered body cannot close
over. Resolution: thread `target` to every body and let the macro emit `k = index(target)`
as an ordinary binding when the declaration names an index. It stays out of the user-facing
slot list; writing `k` is how you ask for it. This mirrors v6, which injects `k = on[2]` at
macro expansion.

**2. Incoming annotations have no declared route.** The mixture switch rule needs the log
scales that *arrived* with its messages. But `args` holds message **data** — the declaration
`m[:μ]::PointMass` is about the distribution, and the body calls `mean(args.m[:μ])` — while
`ann` is an output sink the rule writes to. So an annotation that arrives with a message has
nowhere to go. Recommendation: a parallel accessor keyed exactly like `m`,
`args.ann_in[:out]`, kept out of dispatch, because an annotation must never select the
mathematics. Open item #12 should carry this. *(Not adopted: at the Phase 3 sign-off `ann`
became a two-way slot instead, and the node moved into `ctx.node`, §3.16.)*

**3. The delta-layout collapse is real but partial.** See below.

#### The fallback contract

Resolution becomes a **separate, total function**. `find_rule` returns a `RuleSpec` or a
`RuleNotFound`; it never throws and never runs anything. The fallback is consulted on the
`RuleNotFound` branch only — which is decided *before* any body runs. There is therefore no
`try` anywhere near the body, and an exception from inside a selected rule cannot reach the
fallback even deliberately. A `try`/`catch` around execution would get this wrong, and get it
wrong silently, by turning a broken rule into a missing one.

This applies uniformly to message rules, marginal rules and average energy, which **removes
v6's asymmetry**: today `rule` returns a `RuleMethodError` sentinel while `marginalrule`
throws, so marginal rules cannot have a fallback at all, for no stated reason.

#### Open item #4 — the ruleset axis: DEFER

A downstream package that wants its own rule for a *standard* node and edge declares its own
algorithm and gets it, with no shadowing and no ambiguity, because the algorithm is part of
the signature. Demonstrated in the spike. The piracy argument for the axis was already dead
(§5). Nothing in tree needs scoped rule tables, the fallback contract above is specified
independently of the axis as `PHASES.md` required, and adding the axis later is a new
keyword rather than a resurfacing.

#### The delta layouts — hypothesis holds for input selection, and only for that

Written out as declarations, the four layouts differ in exactly one respect: which messages
and marginals each of the four slots (`q_out`, `q_ins`, `m_out`, `m_in`) consumes. The ~10
engine calls each layout makes are the same calls with different arguments — repeated
wiring, as the hypothesis guessed.

**Three things do not fit, and none is a dependency choice:**

- **Static gating.** `with_statics` (`delta/layouts/default.jl:22-44`) wraps every outbound
  stream in a `combineLatest` against const/data inputs, so the node *waits*, while their
  values reach the rule out-of-band through the function proxy (`FixedArguments.fix`,
  `delta.jl:184`). Measured on the live engine: **0 emissions before the static input
  arrives, 2 after.** A declaration that only names inputs cannot express this.
- **The `N === 1` compile-time branch** (`default.jl:321-327`), substituting
  `of(Message(nothing, true, true))` for an empty group. A statically-arity'd group selector
  covers it only if the zero-arity case is a declared value rather than an empty tuple that
  stalls the `combineLatest`.
- **`q_out` aliasing** (`default.jl:47-64`), which connects the local marginal straight to
  the connected variable's marginal stream. Topology, not a rule input — nothing computes it.

**Consequence.** The collapse is real but partial: dependencies absorb the input selection,
and those three need explicit support in `MessagePassingRulesBase` or they land back in the
engine. `PLAN.md` § CVI projection made `CVIProjection`-as-an-extension conditional on the
collapse; the condition is met for the rules half, provided the three are lifted out of the
layout and into the language.

One more thing the runs settled: **layout in v6 is a function of `(method, inverse)`, not of
`method` alone**, so `method` and `layout` are already two axes. The new design collapses
both into one algorithm *value* with the inverse as a field, after which
`Linearization{Nothing}` and `Linearization{<:Function}` select different rules by ordinary
dispatch — which the hand-written rules 7 and 8 confirm. *(Named in §3.20's terms, that value is
Delta's own algorithm holding the method and the inverse, as v6's `DeltaMeta` did; the
approximation itself, `Linearization` or `Unscented`, is a utility inside it, not the
algorithm. The base tests' toy is now `ToyDelta{I}`.)*

#### Context service contracts (open item #12)

Both hard cases were demonstrated as standalone calls, with no graph and no Rocket:

```
product : (left, right) -> (dist, logscale::Real)
nodefn  : (ctx, target) -> a callable of the FREE arguments only
linalg  : (matrix)      -> a factorisation object (replaces the global cholinv calls, 48 lines in src/)
rng     : ()            -> an AbstractRNG owned by the caller
```

Neither carries an engine type. `product` is the whole of what `rules/mixture/switch.jl`
reaches into the engine for today — the single leak in the rules tree. `nodefn` is what a
delta backward rule towards `in_k` needs: every other input pinned to its current value, one
free argument left, delivered as a callable rather than a node type.

*(Revised at the Phase 3 sign-off, §3.16: `nodefn` is not a service. The node itself sits in
the context as `ctx.node` and the rule calls `getnodefn(ctx.node, …)`; the callable-of-free-
arguments shape is what that call returns.)*

### 3.16 The Phase 3 sign-off

The open items gating the API freeze were brought to the user as an entry brief (evidence
from the code, plus a proposed position each — `PHASES.md` § Phase 3). Outcomes, and the
reasons where the user changed or rejected the proposal:

- **#3 and #9 accepted as proposed.** Group selection is keyed per target; factorisation-
  dependent selection is a distinct algorithm *(refined in §3.20: the default scheme already
  follows the factorisation; only a node that ignores it declares its own)*. Consumed beliefs and the scored partition are
  separate declarations, and a joint must list its members in interface-declaration order,
  rejected otherwise rather than permuted.
- **#10 accepted and made stricter (user).** The proposal said published results are owned
  snapshots and reuse is opt-in per edge. The user's framing is sharper: preallocated
  buffers and messages are **engine internals**, and whether the engine reuses the storage
  behind a message is its own business, **deliberately unspecified** — it may or may not
  happen. So the contract is placed on the *outsider*: anything outside the engine that keeps
  a message must copy it, and a getter the engine offers to outsiders copies by default.
  `InputArgumentsAnnotations`, which today stores references to inputs and results, must
  deep-copy them. This is simpler than specifying eligibility, because it promises nothing
  that would later constrain the engine.
- **#11: the static capability table was rejected (user).** v6 already solves this. The host
  defines `is_delta_node_compatible(method)`, `Val(false)` by default, checked when the method
  is attached (`DeltaMeta(; method)`); a method whose implementation lives in an extension
  gets a specialised error in the host naming the package (`cvi_projection.jl:138-140`), and
  the extension flips the trait. That *is* capability metadata available in the loaded host —
  a registry-level table would duplicate it. Two follow-ups carried into the move: the error
  must also name the method to switch to, and the check belongs in an inner constructor,
  since the positional `DeltaMeta{M, I}(…)` bypasses it.
- **#12 changed (user).** Two simplifications:
  - Incoming annotations do not get a parallel `args.ann_in` accessor. The existing `ann`
    slot carries both directions — read `ann.m[:out]`, write `annotate!(ann, …)` — which
    was the user's choice among three options (one two-way slot; a read-only `ann` plus a
    separate writable sink; a read-only `ann` with outgoing annotations returned alongside
    the result).
  - The **node moves into the context** as `ctx.node`. Nothing dispatches on it, so a slot
    was never needed; the body slots shrink to `(output, algo, ctx, args, ann)`. It follows
    that `nodefn` is not a service at all: `getnodefn(ctx.node, …)`.
  The remaining services are `node`, `product`, `linalg`, `rng`. The missing-input path
  (skip body and post-rule processors, as v6) was left proposed at the sign-off, and
  confirmed when Phase 3 closed (the `execute_rule` docstring).
- **#13 parked (user)** — "I will sleep over it". The brief's `approx_cholinv`/
  `approx_cholsqrt` protocol owned by the approximations package is neither accepted nor
  rejected. Until it is settled, the `linalg` context service is not frozen.
- **RNG accepted as proposed.** `ctx.rng`, owned by the caller; an algorithm holding its own
  RNG is `pure = false`.

Three more decisions were taken while planning Phase 3's execution:

- **`Message`/`Marginal` stay in the engine** (user's choice among engine / base / defer).
  Once the node moved into `ctx.node` and annotations into `ann`, a rule no longer touches
  the envelope at all: `args` holds raw distributions. Putting the envelope in the base
  package would only move engine concepts down a layer.
- **The full interactive surface is Phase 3 scope** (user's choice over minimal or none),
  including the coverage matrix and HTML display.
- **No symbol is formed at run time (user).** The assistant proposed keying a joint by an
  internal `Symbol("y,x")`, on the argument that a comma cannot occur in an identifier. The
  user rejected it: constructing a symbol at run time is slow and must never be relied on,
  and the design had already said `q[:y, :x]` lowers to `getindex(q, Val((:y, :x)))`. The
  key is the member tuple carried in the type, resolved at compile time.

  **Measured at Phase 3 step 2, on 1.10.12 and 1.13.0.** In a `const` lambda behind a
  function barrier, `args.q[:y, :x]`, `args.q[:a, :b, :c]`, `args.q[:p][k]` and
  `args.m[:μ]` all infer concretely, allocate 0 bytes and are JET-clean. The typed code for
  the joint is three `getfield`s ending in `getfield(q.joints, 2)` — constant propagation
  turns the literal symbols into a static `Val`, and the `@generated` lookup resolves it to a
  position. The key-presence check on single keys folds away. The negative control, a
  body receiving the symbols as runtime values, infers a non-concrete type, so the gate can
  fail. The fallback spelling `args.q[Val((:y, :x))]` is therefore not needed.

  One thing Aqua caught on the way: an outer `Messages(::NamedTuple)` constructor silently
  overwrote Julia's implicit default one, which is an error under precompilation. The
  in-process tests had passed regardless, because the package loaded without its image;
  only `persistent_tasks`, which loads it in a fresh precompiling process, failed. Parametric
  containers here declare an explicit inner constructor for that reason.

- **Found at Phase 3 step 6: default-algorithm inheritance reintroduces a load-order
  constraint, but only for the rules that use it.** A rule that omits `algorithm` must
  dispatch on the node's default algorithm type, and that type exists only once the node is
  declared. The macro no longer queries anything at expansion, so the constraint is the
  ordinary one of evaluation order — the signature evaluates `typeof(default_algorithm(node))`
  when the method is defined — and it is satisfied naturally in every package layout that
  occurs: a node's own package declares it before its rules, and a downstream package or
  extension loads after its host. A rule that names `algorithm` has no ordering constraint
  at all. PLAN's claim that definitions "may appear in any order" holds for the latter only,
  and is corrected here rather than in the plan's wording elsewhere.
- **The dependency language, signed off at Phase 3 step 7** (the user's choices on the
  assistant's proposals). Selectors are written `q[:p][k]` (aligned), `m[:in][!k]` (all but
  self), `m[:in...]` (all) and by omission (none), with `select_group_members(f; arity)` for
  a custom one whose arity is checked on every resolution. Static gating is a node-level
  policy, `static_inputs = :fold`: which inputs are static is known only from the graph, so
  the base package records the policy and the engine folds and waits. A selection of no
  members is an empty tuple the engine must treat as satisfied, which removes delta's
  `N === 1` branch. *(As built in case (d): it takes no stream, and the rule receives a tuple
  of `nothing`s, `(nothing,)` for one member; §3.25.)* And `q_out` aliasing needs no representation at all: v6's generic
  clusters already give a singleton cluster the variable's own marginal stream
  (`clusters.jl:116`); delta needed its own copy only because its layout bypassed clusters.
- **Naming (user).** The dependency declaration is `DependenciesSpec`, beside `RuleSpec` and
  `NodeSpec`, and gets the same rich display in step 9. Generic names were replaced by
  descriptive ones across the package — `select` became `select_group_members`, the selector
  types `AllGroupMembers`/`AlignedGroupMember`/`AllGroupMembersButSelf`/`CustomGroupSelector`/
  `SingleInterface`, and `edge`/`index`/`members`/`partition`/`statics`/`groups`/
  `dependencies` became `target_edge`/`target_index`/`cluster_members`/
  `free_energy_partition`/`static_inputs`/`interface_groups`/`dependencies_spec`. The trait
  names GraphPPL and RxInfer already use — `interfaces`, `sdtype`, `nodefunction`,
  `alias_interface` — are kept.
- **What `args` holds for a partly selected group (user, Phase 3 step 8).** The spike's canary
  declared `q[:p...]` and read `args.q[:p][k]` while its dependency selected only `q[:p][k]` —
  a contradiction nobody had noticed, because the spike built its own inputs. The decision: a
  rule declares exactly what its dependency selects, in the same spelling, and the group
  arrives as a full-length tuple in member order with `nothing` where the selection leaves a
  member out. The alternatives were a compacted tuple, where `[k]` stops meaning member `k`,
  and passing the whole group while the selector only schedules, where members the update
  did not wait on may be stale. `!k` cannot be written as a runtime index in a body — `!` is
  not defined on integers, and defining it would be piracy — so an all-but-self body reads
  the tuple with its `nothing` in place.
- **Invocation names mirror definitions (user).** `@call_rule`/`@call_marginalrule` became
  `@call_message_update_rule`/`@call_marginal_update_rule`, beside `@call_average_energy` —
  which v6 never had — and the three `@which_*` queries. This reverses PLAN's earlier "keep
  invocation names short for the REPL"; the user preferred that nothing be generic and that
  each invocation read as the counterpart of its `@define_*`.
- **The devirtualization gate, re-run through the real macros at Phase 3 step 11**
  (`gate:routing-macros`), inside functions with `const` inputs:

  | | 1.10.12 | 1.13.0 |
  |---|---|---|
  | call site reaching one rule | **0**, `Float64` | **0**, `Float64` |
  | indexed target, `k` bound | **0**, `Float64` | **0**, `Float64` |
  | call site reaching two rules (algorithm chosen at run time) | 16, `Float64` | 0, `Float64` |
  | negative control, a body that allocates | 96 | 96 |

  Better than the spike on the two-rule site, which measured 48 bytes and `Any` on 1.10: a
  run-time choice between two algorithm *values* is a small union, which Julia splits, and
  each branch then resolves to one rule statically. JET reports nothing on the one-rule and
  indexed routes. The generated adapters — slot selection, the `k` binding — cost nothing.
- **`preallocate` receives the target** (`(algo, ctx, args, target)` in the lowered form),
  so an in-place rule towards a group member can size its buffer by `k` exactly as its
  body can. The first cut raised an error in that case instead; it was fixed before
  commit.

### 3.17 Phase 4 — what building the test tooling found

`MessagePassingRulesTestUtils` was built in six steps (`ed5cacbb`..`2c78b2b1`): table macros,
registry-backed coverage, node-definition verification, derivative checks and the migration
checker. The findings worth keeping:

- **The table macros.** Three things found while building them:
  - Julia's `Test` can record a result against a *given* source line (`Test.do_test` with a
    `Test.Returned`), identical on 1.10 and 1.13. So a failure reports the user's
    `@test_message_update_rule` line with a sentence describing the case, and the
    caller-splices-`@test` coupling that forced v6's callback form is gone. It is an internal
    of `Test`, used in exactly one function so a change there is one edit.
  - Measuring a rule's allocations must go through the public entry points with the node's
    type specialised (`node::N where {N}`). Through the spec's `::Function` field, or with the
    node passed on as a `DataType`, the call turns dynamic and 1.10 reports 16 bytes the rule
    never allocated — the same trap the Phase 0 spike recorded.
  - The promotion contract is v6's: an output carries the promoted float type of *all* its
    inputs. It caught a toy marginal rule returning `(mean(out), mean(μ))` unpromoted, which
    is exactly the class of rule that breaks `ForwardDiff.Dual` propagation.
- **Node-definition verification found a wrong v6 rule.** Verifying eleven v6
  rules against their own node definitions, ten pass, and one does not:
  `NormalMeanVariance(:μ)` under VMP with a non-point-mass `q_v` returns variance `E[v]`, where
  naive VMP — `exp E_q[log N(out | μ, v)]` — gives `1/E[1/v]`. With `q_v = InverseGamma(3, 4)`
  that is 2 against 4/3, and the log-ratio to the reference varies by 0.68 over the test points.
  The corrected formula verifies to 1.5e-8, and so does the analogous `(:out)` rule. The node's
  own average energy already uses `mean(inv, q_v)`, so v6's messages are not the optimum of the
  free energy v6 computes. The same `mean(q_v)` appears in nine places across `mean.jl`,
  `out.jl` and `marginals.jl`; v6's tests never pass a non-degenerate `q_v`, which is why a
  golden-value table could not see it. Reported as **ReactiveMP.jl#669**, pinned in
  `KNOWN_V6_FINDINGS`, and ported as a declared `:correction` — in the event, in Phase 4.5
  step 3, when NMV was ported. This is the
  outcome Phase 4 was built to produce: a test of the mathematics, not of what a rule returned
  the day it was written.
- **Scale is only meaningful once shape holds.** A rule with the wrong shape has no single
  log-ratio to compare with its log scale, so checking both reported one mistake twice.
  Scale is now checked only after shape passes.
- **Differentiable rules must declare abstract element types.** An in-place rule declared over
  `Vector{Float64}` cannot receive `ForwardDiff.Dual` inputs at all; the derivative check
  surfaces it as a type mismatch with the near miss named. Phase 5's ports should declare
  inputs over abstract element types wherever derivatives are expected to pass.
- **Coverage records the selected rule.** With a broad and a specific rule for the same edge,
  a case covering the specific one leaves the broad one reported — the property PLAN required
  so that a fallback cannot conceal an untested specialisation.

### 3.18 Phase 4.5 restructured — no bridge, the real engine

Recorded 2026-09-23, after the post-Phase-4 audit. Phase 4.5 had been framed as an
*integration slice*, and its first planning question was where the slice lives: either a
bridge inside today's `src/` letting v6's `MessageMapping` call base-package rules
alongside v6 ones, or a throwaway engine that proves the interface and nothing else. The
bridge was the assistant's recommendation. **The user rejected both**: break properly, and
write the real code.

Looking closely, the bridge buys less than it appears to:

- Of the four slice cases, the mixture and the delta node run through exactly the code
  Phase 7 was to demolish — the mixtures' custom `activate!` with `ManyOf` and the
  `reverse(...)` trigger wiring (`normal_mixture.jl:176,183`), and the delta layouts, which
  reach rules through their own path (`delta.jl:301-348`). Bridging them means writing
  adapters for code that is being deleted, and proves the new rules against the *old*
  engine, which is not the question.
- The one thing a live v6 engine could offer — comparing scheduling and free energy against
  the real thing — is unavailable anyway: v6 and v7 share a UUID and cannot load in one
  process (`PLAN.md` § Repository layout). The comparison is fixture-based either way.
- What the bridge kept was a green `main` suite, and that was not a requirement: the branch
  is used by a small internal group and verified locally until the release.

Decided:

- **Phase 4.5 absorbs the start of Phase 7.** It opens with the engine design session that
  Phase 7 lacked, then builds the real v7 engine in `src/` for the slice's nodes. Phase 7
  shrinks to completing it. Phase 5 ports rules straight into the new engine.
- **v6 is a fixture source only.** Before anything is deleted, `compat/v6-comparison`
  records free-energy trajectories, posteriors, log scales and emission order from full v6
  runs (via an RxInfer compatible with ReactiveMP 6.5.0). The v6 engine is deleted when the
  new one lands; v6 rule directories and their tests are deleted as Phase 5 ports them.
  `src/` never holds two engines, and the suite shrinks rather than going red.
- **ReactiveMP takes a hard `[deps]` entry on `MessagePassingRulesBase`**, not a weakdep and
  an extension: `[sources]` on 1.11+, developed at test time on 1.10, as TestUtils already
  does. The cost is a develop step in `ci.yml` and the Makefile.
- **Downstream breakage before the release is accepted.** `IntegrationTest.yml` is not a gate
  for this branch; the coordinated downstream CI remains the Phase 8 requirement.

*(Refined in §3.19: "the real engine" means the v6 engine refactored in place. Its reactive
machinery is kept, and what "deleting the v6 engine" removes is the replaced rule-call,
node-creation and per-node activation paths.)*

The consequence to watch: the design Phase 7 was flagged as missing — group stream wiring,
the `Message` envelope (its mutability now decided by benchmark), edge order in clusters, `EdgeLabel.index`, buffer-reuse
eligibility — is now due at the *start* of Phase 4.5 rather than after Phase 6. That is
the point: those are the decisions the rule interface most needs proven against.

### 3.19 The Phase 4.5 design session

Three read-only investigations fed it, of RxInfer 5.5.2's use of ReactiveMP, the v6 engine's
data flow and retainers, and what the four slice cases need. The outcome is the design brief
in `PHASES.md` § Phase 4.5; what matters for later readers is why.

- **An evolution of the engine, not a new one.** The investigation mapped how deep RxInfer
  reaches into v6: variable constructors, three positional activation-option structs, `score`
  streams of `CountingReal`, and a force-marginal plugin that walks node internals
  (`reactivemp_force_marginal_computation_plugin.jl`). The assistant first read the user's
  "RxInfer will also be refactored" as licence to drop that whole surface. **The user
  corrected the framing:** v7 replaces how rules are found, fetched and called, and how nodes
  and rules are defined and created. The underlying machinery — streams, variables, the
  equality chain, products, deferred messages, scores — is kept. The change is an
  improvement of the rule-call behaviour plus a clean-up. So RxInfer's major release adjusts
  where node and rule creation breaks, and is expected to be small elsewhere. The slice's
  tests build graphs through the engine's own API; RxInfer 5.5.2 records the v6 fixtures.
  See §4, item 22.
- **Rocket stays.** An explicit scheduler was the alternative. Rocket keeps v6's emission
  order comparable against fixtures, and the package table already assumed it.
- **Deferred messages materialise as in v6, by requirement.** The engine investigation
  flagged `DeferredMessage` as a staleness trap: it holds source observables, not values, so
  a message retained and materialised later reads the current values. The assistant proposed
  snapshotting inputs at emission. **The user rejected it: the lazy materialisation is
  load-bearing for the correctness of reactive message passing.** The engine reproduces it,
  and the retained-value guarantee starts at materialisation. See §4, item 21.
- **The slice's rules go into the real packages.** They would have to be ported anyway, and
  a throwaway port is work Phase 5's tool would repeat. About 45 rules were estimated; 54
  were ported (43 message rules, 2 marginal rules, 9 average energies).
- **A cluster over a whole group.** Delta's `q_ins` is a joint over its `:in...` group, and
  the base package rejected any cluster containing a group. The choice was between extending
  the base package, which is general and makes Delta its first customer, and a Delta-specific
  synthetic interface, which is the kind of special-casing the rewrite removes. The user chose
  the extension.
- **`Message`: mutable or immutable is measured, not argued.** The assistant proposed an
  immutable `Message{D, A}`. The user pointed out that v6's `mutable struct` with `const`
  fields is deliberate: a mutable struct is passed by reference, which can avoid copying. The
  typed annotations stay; the struct kind is decided by benchmarking both representations.
- **Signed off.** The user accepted the rest of the brief as written (2026-09-23).
- **`towards` becomes `target`.** Reading the marginal-rule spec, the user found `towards`
  the wrong word: a marginal rule's `(:out, :μ)` is a cluster the rule computes, not an edge
  anything is sent towards. `target` fits both kinds of rule and matches the types already
  used (`Target`, `IndexedTarget`, `ClusterTarget`). It was renamed everywhere, the design
  documents included; earlier text quoting `towards` now reads `target`. Plain English such
  as "the message towards `:out`" is kept where it describes a direction.
- **`FactorizedJoint` alone cannot return a split cluster.** Proposal 5 said a marginal rule
  may return BayesBase's `FactorizedJoint`. Building it showed the gap: the joint is
  positional, so it cannot say which members each block covers. v6's partial splits such as
  `(out_μ = MvNormal, v = m_v)` would be lost, and v6 recovers them only by splitting `out_μ`
  on `_` (`score/score.jl:14-60`), which is unsafe once names may contain underscores. The
  user chose member-tuple labels, and then that the labels *wrap* a `FactorizedJoint`
  rather than duplicate it. `FactorizedCluster` holds the labels, and the joint remains the
  distribution, with BayesBase's entropy and float-type conversion. Upstreaming labels
  into BayesBase was the alternative; it would put a BayesBase release on Phase 4.5's path.
- **Log scales: preserve, do not fix.** Recording the fixtures found three gaps in v6's log
  scales (`PHASES.md` § Phase 4.5, Step 0). The user's position: log scales are
  underdeveloped, niche, and used by a few research papers. The rewrite keeps v6's
  behaviour, takes easy wins only, and leaves a proper fix to a later milestone with its own
  plan and discussion. The recording itself deviates from nothing: an earlier draft wrapped
  `LogScaleAnnotations` to patch one gap, and it was dropped once the fixtures could be
  recorded unmodified.
- **What the investigation found, and the proposals answer.** Clusters and group indices in
  v6 are positions in a flat interface list, re-derived from neighbour order, never from
  GraphPPL's `EdgeLabel.index` (#7). Five node types override activation. `getnodefn` is
  promised by the base docs but defined nowhere. Marginal rules return NamedTuples that split
  a cluster, which v7's single-valued `ClusterTarget` cannot express. And no ReactiveMP test
  runs more than one pass or computes free energy: every trajectory v6 has ever produced came
  through RxInfer.

### 3.20 One DefaultAlgorithm — `BP` and `VMP` were a misreading

Found by the user while reading the tests, before step 4: the base package shipped two
built-in algorithms, `BP` and `VMP`, the node default was `BP()`, the standard nodes declared
`algorithm = BP` or `VMP`, and a test modelled a structured factorisation as a `Structured`
algorithm. That contradicts both v6 and §3.1's own intent. There is **one** algorithm, Bethe
free energy minimisation. Belief propagation, variational message passing and their
structured forms come from the *factorisation*, through the engine's default dependency
scheme. `algorithm` replaces `meta`, which was a rule selector plus a parameter bag. So a
custom algorithm exists only as a rule switcher or as a node's own algorithm (Delta,
Autoregressive, the mixtures). The slip happened because "algorithm" invites naming schemes,
and the early examples did exactly that; nothing forced it.

Decided (user):
- **`DefaultAlgorithm()`** is every node's default; rules omit `algorithm` almost always.
- **Two kinds of custom algorithm.** A direct subtype of `AbstractAlgorithm` stands alone. A
  subtype of `DefaultAlgorithmExtension` inherits the default rules and dependencies for
  whatever it does not define.
- **An inherited rule's `algo` is `DefaultAlgorithm()`**, the algorithm it was written for,
  not the extension's value.
- **`NormalMixture` runs under its own standalone `NormalMixtureVMP`**, documented as always
  variational whatever the factorisation: its rules consume marginals only, as v6's did.
- **Delta's algorithm** is a Delta-owned value holding the method and the optional inverse,
  like `DeltaMeta`; it is built with case (d). *(Built: `DeltaApproximation`, §3.25.)*

**Why inheritance is a second lookup and not subtype dispatch.** The obvious design would let
default rules dispatch on an abstract supertype that extensions subtype. But then an
extension's rule with *broader* inputs than a default rule for the same edge would be more
specific in the algorithm and less specific in the inputs. Julia reports that as a method
ambiguity, and resolution must never throw. Instead, the untyped `find_*` and
`dependencies_spec` fallbacks, which returned `RuleNotFound` or `nothing`, retry with
`DefaultAlgorithm()` for an extension. The extension's own typed rules always win, and no
ambiguity is possible because `DefaultAlgorithm` is concrete and a supertype of nothing. The
routing gate measures the fallback call: inferred, JET-clean, 0 bytes on 1.10 and 1.13. The
test `algorithm:extension-inherits` pins the broader-override case.

It also settles two older open points. #3's "a distinct algorithm for factorisation-dependent
selection" was only ever needed for nodes that ignore the factorisation. And an extension is a
one-level version of #4's `Overlay(mine, standard)`, needing no new keyword.

### 3.21 `Require*FunctionalDependencies` are deleted, not ported

The user asked whether the three types are needed at all. In v6, Probit defaults to
`RequireMessageFunctionalDependencies(in = NormalMeanPrecision(0, 100))`, and
ContinuousTransition defaults to `RequireMarginalFunctionalDependencies(a = nothing)`.
Mixture's `RequireMarginal` path is dead code. RxInfer documents
`where { dependencies = RequireMessageFunctionalDependencies(…) }`, and its Probit, binomial and
multinomial regression tests use it. So they are used, and they bundle three things:

- **what a rule consumes**, which belongs on an algorithm. Both in-tree users already have
  one, as the user pointed out: Probit's `ProbitMeta(p)` drives its moment matching and
  ContinuousTransition's `CTMeta(transformation)` its transformation. Each becomes the node's
  own algorithm, declaring its dependencies;
- **a per-model override**, which becomes selecting an algorithm for the node, a
  `DefaultAlgorithmExtension` with different dependencies (§3.20);
- **an initial value** for a rule that depends on its own edge, which is initialization.
  Decided (user): dependencies stay pure declarations. A node may declare a default initial
  message on its definition, separately, so Probit still works out of the box.

So the types are deleted. Documentation (user): one page explaining how to declare
dependencies, written only in the new terms, and a `MIGRATION.md` section that maps the old
types onto these three pieces. Both are written in Phase 5, with Probit and ContinuousTransition
ported. *(Both were written in Phase 5 step 9, the section in the guide `docs/src/migration-guides/v6-to-v7.md`
(§3.36); Probit and ContinuousTransition are ported in Phase 6.)*

### 3.22 Clean cut, Julia 1.13 only

Two ground rules set by the user before step 4.

**A clean cut, with no transition stage.** The recommendation had been a per-node dual path:
nodes with a `NodeSpec` take the new route, the rest keep v6's `rule()`, and v6's
NormalMeanVariance stays until Phase 5, because v6's GCV rules call it and the v7 port covers
only the slice's rules. The user rejected that: "we delete the old stuff now and do a clean
cut; I don't care about the transition stage." In step 4 the engine keeps only the new rule
path and node creation, and every unported node stops working. Its code and tests are not
deleted, though (user): they move to `legacy/v6/`, mirroring their old paths, never loaded
and never tested, as the reference Phase 5 ports from. Nothing outside the branch uses it,
and downstream breakage was already accepted (§3.18), so a dual path would only be code that
has to be removed later.

**Julia 1.13 only, for now.** The 1.10 floor had cost a develop-at-test-time step for every
unregistered sibling, a separate `test/Project.toml` for a package that needs the test
tooling, and a CI matrix to match. The user's decision is to target 1.13 and wire everything
with `[sources]`, which 1.11+ honours; the floor and its workarounds are reconsidered at
registration. No CI runs without a PR either, so all work is verified locally, and the
workflow files are left as they are until then.

### 3.23 Step 4 built: what it settled, and what it left (2026-09-23)

Step 4 made the clean cut and ran case (a), belief propagation over `NormalMeanVariance`,
through the new engine: `bp_iid`, `bp_iid_missing` and `bp_chain` agree with v6 call by call,
log scales and free energy included. What was decided while building it:

- **The engine refuses what it cannot wire yet, rather than guessing.** `activate!` knows only
  the default dependency scheme: the messages inside an interface's own cluster, and the
  marginals of every other cluster. A node whose algorithm declares its dependencies
  (`NormalMixture` under `NormalMixtureVMP`), or that has an interface group (`:m...`), is
  refused with an error that says so. Wiring the default scheme in their place would hand a
  mixture's rules the wrong inputs and fail somewhere far from the cause. Case (c) lifts both
  refusals. `bethe_free_energy` likewise scores over the factorisation; a declared partition
  arrives with declared dependencies.
- **`Message` stays a `mutable struct` with `const` fields**, by benchmark (the user asked for
  one, §3.19): through the equality chain it was about 10% faster and 40% lighter than an
  immutable struct. On a BP chain the immutable one was 6–8% faster but still 25% heavier.
  Numbers in `PHASES.md` § Phase 4.5, step 4.
- **Typed annotations (`Message{D, A}`, brief item 3) were not built.** Step 4 changed how rules
  are called, not the envelope, so `AnnotationDict` stayed, one change at a time. The
  retained-value test pins that nothing mutates one after materialisation. It belongs with the
  log-scale milestone, which reopens annotations anyway.
- **`factornode` takes names, never positions**: `(name, variable)`, `((group, k), variable)`,
  and a factorisation of the same keys. Brief item 1 asked for exactly this; the engine sorts
  everything into declaration order, so caller order cannot leak into clusters or emissions.
- **`@logscale` went to `legacy/`.** It expanded to v6's rule-scoped `getannotations()`; a rule
  writes `annotate!(ann, :logscale, value)` (already the plan, §3.14).

**The registry, clarified (user, afterwards).** The user asked whether the per-module rule
registry scopes rules, and whether rules from other packages should land in one default
registry, perhaps with a registry keyword and a registry-parametrised `find_rule`. The
answer settled it, and the user decided to **keep the design as it is**:

- **Lookup is Julia's method table, and it is already global.** A rule definition adds a
  method to the base package's `find_message_rule` (or `find_marginal_rule`,
  `find_average_energy`). Every rule, from any package or the REPL, joins that one table when it
  loads, and resolution is ordinary static dispatch. Nothing about lookup is per module.
- **The per-module `__message_passing_registry__` is introspection only**: `list_rules`,
  `check_rules`, `check_rule_ambiguities`, coverage and the near-miss error text. The engine
  never looks rules up in it. It is per module because a package that `push!`es into another
  package's `const` during its own precompilation saves the entry only in its own image
  (`PLAN.md` § Registry); `registries()` stitches the shards back into one view.

Two ways to let users bring their own set of rules were sketched and **not taken**. They are
recorded here for when a concrete use case appears (open item #4):

- **An extension with an explicit parent**, `AlgorithmExtension{Parent} <: AbstractAlgorithm`,
  with `DefaultAlgorithmExtension = AlgorithmExtension{DefaultAlgorithm}`. The fallback becomes
  "my rules, then my parent's", the same mechanism already measured at 0 bytes, so it stays
  static. It closes the one gap today's extensions have, which is composition: an extension can
  overlay only `DefaultAlgorithm`, never a node's own algorithm such as `NormalMixtureVMP`. It
  needs no new dispatch axis. This is the recommended form if the gap ever matters.
- **A registry dispatch axis**, `find_message_rule(registry, node, target, algorithm, args)`,
  with a default registry and user registries that declare a parent. It can be static (a
  singleton type carried in `MessageMapping`, like the algorithm), and it would separate *whose
  rules* from *which inference*, e.g. `infer(...; rules = MyRules())` for a whole run. But it is
  a second axis doing nearly the algorithm's job. Every lookup becomes registry × algorithm,
  there are more places for ambiguities, errors read worse, every signature grows, and the
  Phase 0 gate would need re-measuring.

Either one would also address the one real hazard of a global table. Loading a package that
defines a `DefaultAlgorithm` rule changes results for everyone, and a second definition of the
same signature silently replaces the first (`duplicate_rules()` reports it afterwards).

### 3.24 Case (c): the mixture, and v6's `reverse(...)` as a schedule (2026-09-23)

Case (c) wired declared dependencies and interface groups into the engine and ran
`normal_mixture`. The first run agreed with v6 for 34 rule calls and then diverged, in order
and in values: after five iterations `z[1]` was 0.79 against v6's 0.99999999. The cause was open
item #6. For `:out` and `:switch` v6's mixture subscribed to `(out, reverse(precisions),
reverse(means))`, while the rule read `(out, means, precisions)` through `map_to`. The engine
subscribed in declaration order, `(out, m₁, m₂, p₁, p₂)`.

**The subscription order is the update schedule.** Posteriors are subscribed lazily, and the
order in which a node's inputs connect their shared marginal streams decides the order in which
messages materialise and marginals update within an iteration. In variational message passing
that decides what each rule reads. Measured on the fixture, each order against v6 at `1e-9`:

| `:switch` subscribes | against v6's five iterations |
|---|---|
| `out, m₁, m₂, p₁, p₂` (declaration order) | diverges at call 35, in order and values |
| `out, p₁, p₂, m₁, m₂` (groups swapped) | free energy and every posterior agree; call order differs |
| `out, m₂, m₁, p₂, p₁` (members reversed) | diverges at call 35 |
| `out, p₂, p₁, m₂, m₁` (v6's) | agrees call by call |

And the free energy per iteration, run for 50:

| iteration | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 50 |
|---|---|---|---|---|---|---|---|---|
| v6, and precisions first | 22.926 | 19.096 | 17.558 | 17.55262 | 17.5526135 | 17.55261352151 | 17.552613521506 | 17.552613521506 |
| means first | 26.856 | 22.519 | 22.318 | 21.246 | 17.638 | 17.55268 | 17.5526136 | 17.552613521506 |

Both schedules decrease the free energy monotonically and reach the same optimum, with
posteriors identical to every printed digit at 50 iterations. Means first spends about six
iterations on a plateau and converges in about 13; precisions first in about 7. The fixture's
five iterations fall inside the plateau, which is why the first run looked so far off.

So the reversal is two things. **The group order, precisions before means, is a real schedule
choice**; it converges faster here, and nothing says it would on another model. **The member
reversal is inert**: the members of a group do not depend on each other within an update, so
it reorders emissions and changes no value.

**Decided (user): no reversal in the engine.** v6 records no reason for it, and it is not
mathematics. The engine subscribes to a target's inputs in declaration order, so a node's
dependency list states its schedule, which is explicit and has a stated reason.
`NormalMixture` declares `:out => (q[:switch], q[:p...], q[:m...])` and
`:switch => (q[:out], q[:p...], q[:m...])`, with the reason in its docstring. The fixture is
compared with `trace_order = :within_iteration`: the same calls with the same results in each
iteration, in any order. A generic engine rule reproducing v6's order exactly (singles first,
then the group members reversed) was built and measured, and then removed.

Also settled in case (c):
- **A group reaches a rule as one tuple**, full length with `nothing` for the members not
  selected. Inputs are labelled by name, cluster tuple or group member, and the members of a
  group fold into a type-level `GroupInputs{name, n, members}`, expanded in generated code, so
  no name is formed at run time. The node score folds its cluster marginals the same way.
- **A declared consumed marginal of one interface is its variable's marginal**, whatever the
  factorisation; a tuple key is a cluster of the factorisation, or an error (#9).
- **A declared free-energy partition must be the factorisation**, or `activate!` raises an error
  naming the algorithm (#9's activation-time check); free energy is then scored over it.
- **#7 is closed on the engine side**: group indices are the caller's `(:m, k)`, never positions.

### 3.25 Case (d): the Delta node (2026-09-23)

Case (d) ran the Delta node through the new engine, on `delta_unscented`, `z := f(x)`, and on a
fixture recorded for the case, `delta_unscented_static`, `z := f(2.0, x, s)` with a constant
and a data input. Both agree with v6 call by call. What was decided:

- **Delta has a package of its own, created early** (user): `lib/DeltaMessagePassingRules`,
  `INVENTORY.md`'s `node:Delta`. Putting it in `StandardMessagePassingRules` would contradict
  the standard/node split, and putting it in the engine would make ExponentialFamily an engine
  dependency. The name follows the sibling pattern, `<family>MessagePassingRules`; the user
  asked for an opinion and may still rename it, which is one mechanical commit before
  registration.
- **The node is a type, the function is the engine's.** `DeltaFn{F}` is declared like any node
  and dispatched on as `Type{<:DeltaFn}`; the engine's `FactorNode` holds the function, given as
  `factornode(…; nodefn = f)`, and implements `getnodefn(node, Target(:out))`. A closure works
  as well as a named function.
- **`DeltaApproximation(; method, inverse)` is v6's `DeltaMeta`.** The known inverse is part of
  the algorithm and a rule reads it from `algo`; `getnodefn` gives only the forward function,
  and the base docstring, which had said it also gave the inverse, is corrected. The
  unknown- and known-inverse layouts are two dependency declarations, on the two forms of the
  algorithm, both with the partition `[(:out,), (:in,)]`.
- **Static inputs are folded by the engine, as v6 did.** Under `static_inputs = :fold` the
  group members connected to a constant or to data get no interface, so they are not connected
  and add no point entropy, as in v6; the rest are renumbered `1:n`, which is why `x` is
  `(:in, 1)` in `f(2.0, x, s)`. A `StaticFold` calls `f` with the latest static values in their
  places, and every update of the node waits for them (v6's `with_statics`). Only a group's
  members are folded, and a folding node has exactly one group.
- **A deterministic node's clusters are `out` and the joint over its inputs**, whatever the
  caller's factorisation, since a deterministic node has no variational factorisation of its
  own. The joint is computed by the marginal rule from the messages on every interface (v6's
  `q_ins`), and the node's free energy is minus its entropy. A joint over a whole group is keyed
  by the group's name, `(:in,)`, even with one member, and is never aliased to a variable's
  marginal. Belief propagation through a deterministic node under the default scheme reads the
  messages on every other interface.
- **An empty selection still reaches the rule.** `m[:in][!k]` with one input selects nothing;
  the rule receives `(nothing,)`, so one known-inverse rule serves any number of inputs.
- Without a `DeltaApproximation`, a Delta node runs under `DefaultAlgorithm` and no rule is
  found; the `RuleNotFoundError` lists the `DeltaApproximation` rules as near misses. v6 raised
  a dedicated error; nothing here needed one.

### 3.26 The Phase 5 entry brief: no transform tool (2026-09-23)

Phase 5 was counted before it was planned. 36 of the 50 rule directories in `legacy/v6/` go to
`StandardMessagePassingRules`: 250 message rules, 82 marginal rules and 38 average energies,
of which Phase 4.5 ported 43, 2 and 9, leaving 207, 80 and 29. About 50 of the marginal rules
return a NamedTuple and become `FactorizedCluster`s.

**No transform tool (user).** PLAN had called for a JuliaSyntax transform, run with v6 loaded
so that `interfaces(fform)` could decide whether `q_y_x` is one interface or a cluster. The
alternative offered was a minimal tool for the mechanical part. The user chose neither: the
rules are ported by hand or by agent. The slice's 54 rules were ported that way, and what
remains is dominated by work a tool would only flag: `FactorizedCluster` returns, algorithm
questions such as the correction strategy, helper extraction, and tests that do not exist in
v6. The gates stay strict, per directory: v6's tables, a `compare_standard.jl` case per rule,
verification against the node definition where it applies, and `check_rules`. `MIGRATION.md`
(the docs page since §3.36) is written from what the ports find, with its pairs as doctests; the requirement that the tool
and the guide derive from one source goes with the tool.

**Rule-to-rule calls become helper functions (user).** About twenty v6 bodies call another
rule with `@call_rule`, which has no in-body equivalent. Calling the interactive
`call_message_update_rule` from a body was the alternative; it would go through rule lookup at
run time, which the purity and devirtualisation gates never measured. A helper holding the
shared mathematics, called by both rules, is static and already the pattern
(`normal_mean_precision_energy`).

The order of work, and the gaps found while counting, are in `PHASES.md` § Phase 5.

### 3.27 A cleanup phase for historical remarks (user, 2026-09-23)

The rewrite writes its own history into the code as it goes: a test says "Phase 5, step 4",
a docstring says what v6 did, a comment points at a section of this document. That is useful
while the work is in progress, since the reason for a choice sits next to it, and it is
noise once the work is done, because git already holds the history. The user asked for one
cleanup at the very end rather than a rule against writing them now: remarks stay while they
help, and **Phase C**, before the release, removes them, keeping any reason a remark carries
in present terms, and removes the working documents themselves. Its criteria are in
`PHASES.md` § Phase C.

### 3.28 The matrix correction is a context service (user, 2026-09-23)

v6 passed MatrixCorrectionTools' strategies as a rule's `meta`: MvNormalMeanPrecision's
precision rule calls `correction!(meta, …)`, with `nothing` as its default, and `*` and `dot`
default to `ReplaceZeroDiagonalEntries(tiny)`. Since `meta` became the algorithm (§3.20), the
question was whether the strategy should be an algorithm, a `DefaultAlgorithmExtension` field
for instance. The user decided it is neither: it is a setting of the context,
`ctx.matrix_correction` (named for precision over `correction`), next to the future Cholesky
strategy of `linalg` and `rng`. It changes numerics, not which rule runs, which is what the
context is for; an algorithm would multiply the rule table by every strategy. `nothing` is the
identity, the engine passes it until Phase 7 lets a user set it per node, and step 7 decides
how `*` and `dot` keep v6's default. The step 5 brief in `PHASES.md` § Phase 5 has the details.

### 3.29 `public_equivalent`: an efficient type's public counterpart (user, 2026-09-23)

v6's Wishart and InverseWishart nodes added `to_marginal(::WishartFast) = convert(Wishart, …)`
and the same for `InverseWishartFast`, so that rules can work in the efficient types while
users see the Distributions ones. `to_marginal` is an engine function, and the engine has no
runtime dependency on ExponentialFamily, so the methods needed an owner. Tracing its call
sites showed it is not about marginals as such: the engine applies it in `as_marginal`, to
every marginal a variable forms, and downstream rules receive the result as well as users.
What it means is "the same distribution, as the public type users expect instead of an
efficient working type"; a marginal is only where the engine applies it today, and callbacks
or fixtures could use it too.

The user renamed it to say that: **`public_equivalent`**, stressing that the result is the
same distribution, not a different one. Names tied to marginals (`marginal_form`), to users
(`user_facing_form`, wrong since rules see it too) or clashing with exponential-family terms
(`canonical_form`, `natural_form`) were rejected. `MessagePassingRulesBase` owns it for now,
the identity by default, documented with the working/public split and its uses; the engine
calls it where it called `to_marginal`, and Standard adds the two Fast Wishart methods when it
ports Wishart (step 6). Its right owner is **BayesBase**, with ExponentialFamily extending it for
the Fast types it defines, so that any package gets it; that move is recorded for Phase 8, the
ecosystem integration, beside the upstream `Uninformative` identity (§3.27). A reverse,
working-form direction is not needed: rules accept both types.

### 3.30 The domain-specific nodes get their own packages (user, 2026-09-23)

§2's *Standard versus models* sent GCV, Probit, SoftDot and GaussianCoupling to one sibling
package, its name deferred to Phase 6. The user decided each gets its own node package
instead, as Flow, BIFM, the Pólya nodes and the transitions do. The placeholder destination
`models` is gone from `INVENTORY.md` and `scripts/inventory.jl`; the four are `node:GCV`,
`node:Probit`, `node:SoftDot` and `node:GaussianCoupling`, and the deferred naming question
closes with it.

### 3.31 An unset matrix correction is the rule's default (user, 2026-09-23)

§3.28 made the matrix correction a context service with `nothing` as the identity, which was
v6's default for MvNormalMeanPrecision. `*` and `dot` default to `ReplaceZeroDiagonalEntries(tiny)`
in v6, so one meaning of `nothing` cannot serve both. The user decided that `nothing` means
**not set**, and each rule falls back to its own default: v6's `default_meta`, which is the
correction for `*` and `dot` and none for MvNormalMeanPrecision, so no behaviour changes. An
explicit identity is MatrixCorrectionTools' `NoCorrection()`. A rule reads the service through
`matrix_correction(ctx, default)`, a helper of `MessagePassingRulesBase` beside the service.
Two alternatives were rejected: a node-level default the engine would put into the context,
which is more machinery for the same effect, and dropping v6's default for `*` and `dot`,
which changes results whenever a precision has zero diagonal entries.

### 3.32 Sampling draws from `ctx.rng`, which the engine supplies (user, 2026-09-23)

Two `*` rules, towards `:out` and `:A` for general univariate inputs, approximate their
messages with 3000 draws from the global RNG. The Phase 3 decision is that randomness comes
from `ctx.rng`, owned by the caller (PLAN § RNG ownership). The engine passed `rng = nothing`,
so the rules would have had nothing to draw from. The user decided the rules declare
`ctx = (:rng,)`, and the engine passes `Random.default_rng()` until Phase 7 makes the RNG an
activation option, so behaviour matches v6. The number of draws stays v6's 3000; making it
configurable is recorded for Phase 7 with the RNG option. `Random.default_rng()` changes its
stream between Julia versions, so the user asked that no test pins a number drawn from it:
the tests pass a `StableRNG`, and the sampled messages are checked against quadrature of their
defining integrals, which is also how the v6 comparison checks them, since two versions'
draws cannot match. Keeping the global RNG with `pure = false`, or leaving the rules unusable
until Phase 7, were rejected.

### 3.33 The arithmetic nodes are the functions themselves (user, 2026-09-23)

v6 declared `+`, `-`, `*` and `dot` as nodes on `typeof(+)` and so on. The base package
needs nothing special for that: `@define_factor_node(node = +,
…)` dispatches on `typeof(+)` through `node_dispatch_type`, as Delta's tests already exercise.
The user decided to keep the functions as the nodes if that works without hacks, and asked
for a fallback otherwise: the node types `Addition`, `Subtraction`, `Multiplication` and
`DotProduct` in Standard, with a base hook `node_type(::typeof(+)) = Addition` that
`factornode` would apply. The port of `+` is where this is checked. The one cost of the
functions is that Standard adds methods on Base's function types, which Aqua's piracy check
lists as owned, as it already does for ExponentialFamily's node types.


### 3.34 The `product` service returns the product's own log scale (user, 2026-09-23)

`Mixture`'s switch rule needs, for each input k, the log scale of `m_out × m_inputs[k]`. v6
built a throwaway `randomvar` and a `MessageProductContext` with `LogScaleAnnotations` to get
it, and the result summed three terms: the two incoming log scales and the product's own
normalisation (`compute_logscale`). Phase 0 had already settled the service's signature,
`product : (left, right) -> (dist, logscale)`; what was open is which log scale it returns.
The user decided it returns **only the product's own**. The service stays pure distribution
algebra over two distributions, and the rule adds the incoming log scales it reads from
`ann.m`, so what it sums is written in the rule. The alternative, the service adding the
inputs' log scales as v6's context did, would have made it take annotated messages rather
than distributions. The engine supplies it from `rule_context`, with v6's `GenericProd`.

### 3.35 Mixture has no average energy, and runs under `MixtureBP` (user, 2026-09-23)

v6's `Mixture` energy logs a warning and returns 0.0 on every call, so a free energy of a model
with a Mixture was silently wrong. The user decided the port defines **no** average energy: a
free energy then raises the base package's `RuleNotFound`, naming the node, instead of a wrong
number. It is a gap, recorded in the v6 → v7 guide. Keeping v6's placeholder was rejected.

The node's rules are sum-product over the incoming messages whatever the factorisation, so it
runs under an algorithm of its own, as NormalMixture does. The sketches called it
`MixtureVMP`, which mislabels it; the user chose **`MixtureBP`**.

### 3.36 Closing Phase 5: the guide, the docs, and what leaves `legacy/` (user, 2026-09-23)

Four questions shaped step 9, and the user decided each:
- **The migration guide is a docs page**, `docs/src/migration-guides/v6-to-v7.md`, beside the
  v5 → v6 one. PLAN had it as `MIGRATION.md` in the base package, but most of what it maps is
  the engine's, Standard's or Delta's, and a docs page is where downstream authors look.
- **Only v7 code runs.** Each before/after pair shows the v6 side as plain code and the v7 side
  as a doctest. The v6 code serves the migration only and is deleted with the refactor, so it
  is run nowhere, in CI or in a second environment; a runner pairing 6.5.0 with the new
  packages was rejected as complication for a side that will not exist.
- **The docs are rewritten for what is ported**: the concept, engine and library pages, and the
  pages the exit criteria name. The pages of nodes Phase 6 has not ported are removed, and each
  node package writes its own with its port; the v5 → v6 guide stays as history.
- **`legacy/v6/` keeps only what Phase 6 ports from.** The v6 engine and rule-system files are
  deleted, being replaced and kept by git and by the 6.5.0 release. v6's rule fallbacks and
  `StandaloneDistributionNode` are not carried over, and the guide says so. The helpers the
  unported nodes still use stay, as those nodes' material.

### 3.37 Log scales stay where they are, experimental, until Phase 7 decides (user, 2026-09-24)

The post-close review found `INVENTORY.md` sending the log-scale and input-argument annotation
processors to the base package, where they are not and cannot go. What each layer holds:
- the **base package** carries annotations generically: a keyed store, `annotate!`, and
  `ann.m`/`ann.q` for those that arrived with the inputs. It gives `:logscale` no meaning;
- **Standard's rules** write `:logscale` by convention, and Mixture's read the incoming ones: its
  switch is a softmax over them, so without log scales Mixture has no rules;
- the **engine** owns the policy: `LogScaleAnnotations` sums them across products and fills in
  zero for point-mass inputs, and `InputArgumentsAnnotations` records the inputs.

The processors hook the engine's `AnnotationDict`, `MessageMapping` and messages, which the
package split keeps out of the base package, so they stay in the engine and the inventory says
so. The key itself is a contract between Standard and the engine that neither owns.

The user weighed four paths: leave the layering and decide later; give the key an owner in the
base package now; move the processors into the base package; or drop log scales in this
refactor. Dropping is tempting, since the feature is niche and underdeveloped, but it breaks
Mixture and contradicts §3.19's "preserve, do not fix". Giving the key an owner now would be
churn, since typed annotations replace it in Phase 7. **Decided:** the layering stays; the
docs mark log scales experimental; and Phase 7's log-scale milestone decides whether to fix
v6's gaps, with typed annotations and the key owned by the base package, or to drop the feature
and Mixture's rules with it.

### 3.38 A node declares what it requires of the graph (user, 2026-09-24)

v6's NormalMixture and GammaMixture constructors checked for at least two components, as many
of each kind, and a mean-field factorisation. The generic `factornode` lost all three, so a
mixture with three means and two precisions was accepted and its switch rule, zipping the
groups, silently dropped a component. Inferring the relation from the dependencies (an indexed
target `(:m, k)` reading `q[:p][k]` implies `length(m) ≤ length(p)`) was considered and not
chosen: the user wanted it **declared**. `@define_factor_node` takes `matched_groups`,
`min_group_length` and `factorisation = :meanfield`, the `NodeSpec` keeps them, and the engine
checks them when it creates a node.

### 3.39 Rule coverage counts a rule a test calls by hand (2026-09-24)

`check_rule_coverage` existed but no suite ran it, so `PLAN.md`'s mechanically enforced
coverage was not. Wired into Standard and Delta after an unfiltered run, it reported 50 of
Standard's rules unselected (Delta had none), and Uninformative and Mixture with no rule
selected at all. 39 of the rules were tested, by `@test call_average_energy(…) ≈ …` rather
than a table, and a direct call recorded nothing; the other 11 had no test, most of them
shadowed by a broader rule their tests reached instead, and got one. Rewriting those tests as tables would add the tables' type
promotion checks to rules written as direct calls precisely because they do not support them.
Instead the base package's interactive calls report the rule they select to observers, and the
test tooling registers one; an engine resolves rules itself and never reaches them. The gate is
then zero gaps, which is also `PLAN.md`'s "fail on decrease".

### 3.40 Phase 6's shape: the split waits, SoftDot stands alone, and the numerics' homes (user, 2026-09-24)

Counting Phase 6 for its brief raised four questions, and the user decided each:
- **The repository split moves to Phase 8.** `PLAN.md` § Repository layout had the monorepo
  split at Phase 6, once the base API froze and the engine interface was proven. Both have, but
  Phase 6 adds ten packages, every one wired with `[sources]`, on Julia 1.13 only, with no CI
  until registration (§3.22). A split now would turn each cross-package change into pull
  requests and dev pins for packages nobody can yet install; at registration, compat bounds
  and CI are set anyway, so the split costs least there.
- **SoftDot does not depend on AR.** Its `y` rule called AR's and it used AR's `ar_slice` and
  `add_transition`. Depending on the AR package was the alternative, and so was putting SoftDot
  into it. The user chose neither: SoftDot's package reimplements what it needs and is tested on
  its own, so the two can change independently.
- **Gauss–Hermite cubature and `approximate_meancov` go to `MessagePassingRulesApproximations`.**
  `INVENTORY.md` sent them to the Pólya package as their only user, but Probit's energy and GCV's
  `ExponentialLinearQuadratic` use them too. The numerics package takes FastGaussQuadrature and
  the part over means and covariances; the methods that take a distribution stay with the node
  packages, so it still depends on no distribution package (§3.9b).
- **The algebra helpers go to `StandardMessagePassingRules`**, not to the numerics package:
  `mul_trace`, `rank1update`, `negate_inplace!` and `mul_inplace!`, which AR, SoftDot,
  ContinuousTransition and DiscreteTransition use, beside Standard's `diageye`. Duplicating them
  per package was the third option.

The order of work, also the user's: the numerics and the deletions first, then the nodes by
dependency, each step briefed before it starts (`PHASES.md` § Phase 6, *Entry brief*).

### 3.41 A declaration can extend the default scheme (user, 2026-09-24)

ContinuousTransition's rule towards `a` reads `q(a)`, the expansion point of its transformation.
v6 declared it with `RequireMarginalFunctionalDependencies(a = nothing)`, which kept the default
scheme and inserted the variable's marginal among the marginal dependencies. Every other target
follows the factorisation, and v6 documents two of them, mean-field and `q(y, x) q(a) q(W)`.
§3.21 moved what a rule consumes onto the node's algorithm. A declared dependency spec, however,
is a fixed list per target, so it can express only one factorisation.

The user chose **auxiliary inputs**. `:a => (default, q[:a])` is the default scheme's inputs plus
`q(a)`, so the node keeps one algorithm, `CTVMP(f)`, for both factorisations. This is what `PLAN.md`
§ Open questions 9 anticipated when it separated what a rule consumes from the partition free
energy is computed over: an auxiliary marginal is consumed and never scored.

The alternatives:
- **Two algorithms, one per factorisation**, each with a full declaration and a declared partition,
  so that a mismatch is an activation error. This needs no change to the base or the engine, but
  a model would name both its constraints and the matching algorithm.
- **The structured factorisation only**, which drops one v6 documents.

The cycle `m(→a) → q(a) → m(→a)` is broken as in v6, by `combineLatest(…, PushNew())`: the message
is recomputed only once every input has refreshed. `check_rules` checks the auxiliary inputs
only, since the rest depend on the factorisation (`PHASES.md` § Phase 6, *Step 5 brief*).

### 3.42 ContinuousTransition linearises the same way in every rule (user, 2026-09-24)

Porting ContinuousTransition showed that its rules approximated `A = f(a)` in two different ways:
- **The rules towards `y` and `x`, the joint and the energy** used the linearisation `Ā`, `f`
  expanded at the mean of `q(a)` plus a standard deviation and evaluated at the mean, with the
  Jacobians `Fᵢ` of its rows for the spread.
- **The rules towards `a` and `W`** took each row as `(Fᵢ a)ᵀ`, linear through the origin.

The two agree for `reshape`, the common case, and there the port agrees with v6. For an affine or
nonlinear `f` the second form drops the offset `f(m_a) - J m_a`; v6's documented rotation example
is one such `f`. A Monte Carlo check with an affine `f` confirmed that the `W` rule's
`E[(y - A x)(y - A x)ᵀ]` was wrong.

The user chose to **correct and declare**: every rule uses the first linearisation, so the rule
towards `a` takes the offset and the rule towards `W` the energy's expectation. The alternatives:
- keeping v6's behaviour, documented, with an issue;
- restricting `f` to linear maps, which breaks the rotation example.

`Ā` and the expansion point are v6's and are not reconsidered.

### 3.43 BinomialPolya's energy is an expectation, and corrections get no issues (user, 2026-09-24)

BinomialPolya's average energy needs `⟨softplus(ψ)⟩` with `ψ = xᵀβ`. v6 meant to take the plug-in
`softplus(⟨ψ⟩)` by default and a Monte Carlo estimate under a meta, but overwrote the estimate with
the plug-in. So it always returned the plug-in, which is biased low since softplus is convex:
1.069 against 1.534 in the case checked. The user chose **Gauss–Hermite** over the normal of `ψ`,
as MultinomialPolya's energy already computes it: deterministic, and needing no generator. Fixing
only the overwrite, and keeping the plug-in as the default, was the alternative.

The user also decided that the v6 errors the ports correct get **no issues**. Each is declared in
its comparison and recorded in `PHASES.md` and the changelog, so an issue would only restate a fix
already made. The Phase 6 entry brief's "each with an issue" is superseded.

### 3.44 Stateless BIFM, and scratch space for rules (user, 2026-09-24)

v6's `BIFMMeta` was a mutable cache. The rule towards `zprev` wrote intermediate quantities into
it, and the other rules read them back. So a node's results depended on the update order, and
one meta shared across nodes corrupted them. Two things were fused there:
- **state shared between a node's rules;**
- **working memory.**

The shared state is removed. Each rule recomputes what it needs, reading its own edge's message
where it read the cache (§3.41), at the cost of a small Cholesky inverse per rule. The algorithm
is immutable and the rules are pure.

The user proposed working memory as a feature of its own, and named it **`scratch`**. A rule
declares how to build it from its inputs and takes it as a body slot. The engine keeps one per
outbound stream and reuses it. Three points shape the design:
- **Write-before-read.** It carries nothing between calls, and the engine may drop or rebuild it,
  so a rule stays pure and its result depends only on its inputs.
- **Per rule, never shared across a node's rules.** Sharing would bring back v6's cache and its
  ordering hazard.
- **Separate from `inplace`.** A rule that returns a fresh distribution may still want scratch,
  and the two combine.

Scratch never leaves the rule, so reusing it cannot meet the retention problem that §3.16's
**#10** left the engine free to avoid for output buffers. That makes it the safe half of buffer
reuse to build first. TestUtils enforces the contract by running a rule again on a reused scratch
poisoned with NaN.

BIFM's free energy failed in 6.5.0 (an `Inf` node bound). It is **out of scope**: the port raises
an error naming the node, and a correct Bethe free energy is a follow-up.
### 3.45 DiscreteTransition is a tensor node (user, 2026-09-24)

DiscreteTransition was bolted onto v6's engine:
- its extra arguments were aliased by position into `T1`, `T2`, …;
- its `rule`, `marginalrule` and `score` were written by hand for any target and any
  factorisation;
- a joint's tensor axes were found by parsing its name as a string, `"in_t1_t5"`;
- about 50 explicit Tullio rules sat beside them.

Enumerating the factorisations as typed rules was considered first. It was rejected because it
cannot cover joints of arbitrary members of the `T` group: dispatch cannot match "`out` with any
`T` member". An untyped escape hatch, v6's hand-written `rule` in the base, was the other
alternative, and it steps outside what `check_rules`, coverage and the table tests rely on.

The user asked for something better, since the node is important, and agreed to this. Every
DiscreteTransition rule is one computation, the tensor `E[log A]` weighted by each input along its
axes, and the engine knows each input's axes statically: they are its cluster key. So:
- **joints of part of a group** are allowed, keyed with the members, `(:out, (:T, 1))`;
- **`default` in a rule's arguments**, the counterpart of `default` in a declaration (§3.41), lets
  one typed rule take whatever inputs the factorisation delivers, with the typed inputs named
  beside it.

The node's rules are then written once, for any factorisation and any number of `T`s. Rules with
explicit inputs remain more specific, so a fast path can be added where a measurement asks for it.
`default` arguments are a general feature of the base, with DiscreteTransition as their first
user.

*Decided while building:*
- **A block of one member of a group, in a `FactorizedCluster`, is a joint of that member**,
  `q[((:T, 1),)]`, not an entry of `q[:T]`. Building the group's tuple needs its length, and the
  cluster's key does not carry it: `(:out, (:T, 1))` says nothing of how many `T`s there are.
  Only a `default` rule reads such a block, walking `rule_inputs`, so nothing is lost.
- **An observed member inside a whole group stays in the joint.** A cluster over every member is
  keyed `(:out, :in, :T)`, and blocks splitting `T` would not partition that key. The joint keeps
  the member as a one-hot axis, which is the same distribution as v6's blocks, and whose entropy
  is the same.
- **No fast path** (the measurement is in `PHASES.md`, step 9's *Progress*): once `rule_inputs`
  folds and a point mass with nothing to sum out skips the logarithm and the exponential, the
  generic rule is within 1.0–1.6 times v6's explicit ones from ten states on.

### 3.46 Phase 7's three decisions (user, 2026-09-25)

The entry brief asked three questions, and the user answered:
- **Log scales are fixed, not dropped.** §3.37 left the choice to this milestone. Dropping would
  have taken Mixture's rules with it, since its switch is a softmax over incoming log scales.
  Fixing them brings typed annotations (`Message{D, A}`) in place of the mutable `AnnotationDict`,
  and gives the log-scale key an owner in the base package. The milestone stays the phase's last
  item, so the rest is settled first.
- **RxInfer is adapted in this phase**, on a local branch of its checkout wired to this branch
  with `[sources]`, rather than deferred to a phase of its own. Its suite and RxInferExamples
  models are the check; nothing is pushed there without asking.
- **The diagnostics are activation options**, checked as each rule is resolved, rather than a
  separate pass over a built graph. The engine knows the resolved `RuleSpec` at that point, with
  its `pure` and `inplace`, and RxInfer forwards the options from `infer`.

### 3.47 The standalone distribution node comes back (user, 2026-09-25)

§3.36 did not carry over v6's `StandaloneDistributionNode`, the node of `x ~ d` for a distribution
*value* `d`. Adapting RxInfer found that its tests and documentation use it: priors passed to a
model as arguments, `θ ~ prior`, and values of families with no node of their own, such as
`Truncated(Normal(0.5, 1.0), 0, 1)`. Rewriting `x ~ d` into the family's own node, the other
option, would not cover those. So it is an ordinary node of Standard, `StandaloneDistribution`,
`out ~ d` with `d` a constant: its message towards `out` is `d`, and its average energy the cross
entropy `KL(q ‖ d) + H(q)`, so the node's free-energy term, the energy less `H(q)`, is v6's
`KL(q ‖ d)`. v6 made it a special engine node type; here nothing in the engine is special but one
method: the entropy of a point mass whose point is a distribution is `−∞` in that
distribution's float type. RxInfer builds the node for a distribution value, and counts its
hidden constant in the free energy's point entropies.

### 3.48 Log scales stay as v6 has them, past the release (user, 2026-09-25)

§3.46 had log scales fixed in Phase 7, with typed annotations and the key owned by the base
package, and item 7's brief proposed a contract (exact belief-propagation rules must annotate
one, verified; the rest record `missing`). **The user set that aside:** this is a transition
release. Log scales reproduce what v6 does, with no new rules annotating them, none removed, and
no new requirement or error; typed annotations are not done, and may not be needed; the mutable
`AnnotationDict` stays. Every earlier decision about fixing them is void, and what log scales
should be is decided after the refactor is released. The brief's survey stands as the record:
22 of Standard's 83 belief-propagation rules annotate a log scale, none elsewhere.

### 3.49 Rule fallbacks back; the context services one option (user, 2026-09-25)

§3.36 had not carried v6's rule fallbacks over, for nothing in the tree needed them; RxInfer's
documentation does. The design had kept their place: resolution is total, so a fallback sits on
the `RuleNotFound` branch only and can never swallow an error from a rule. So they come back as
the activation option `rulefallback`, and the base package provides `NodeFunctionRuleFallback`,
v6's computation from the node function.

The generator was a flat activation option of its own, `rng`, and `matrix_correction` had none.
The user asked for the context to be one option instead, and open: a store of anything a rule may
need, so a rule may declare services of its own and the base does not restrict their names; the
base documents only those an engine supplies by default. `RuleContext` is a mutable object
holding a typed `NamedTuple`, so it is passed by reference and every read of a service is
inferred. The engine builds one per node at activation, its defaults (`node`, `product`, the
task's `rng`) merged with the node's `context` option, which adds or overrides services; RxInfer
forwards the option.

### 3.50 Log scales are first-class; `RuleResult`; no `product` service (user, 2026-09-25)

The user found the `product` context service obscure: a default service of every node,
`(left, right) -> (distribution, logscale)` with `GenericProd`, used by one rule, Mixture's switch,
and returning a log scale the base package did not own. Discussing it reopened log scales
themselves, and supersedes §3.34 (the service), §3.37 (log scales experimental, the key
ownerless) and §3.48 (log scales as v6 has them past the release).

- **What a log scale is** (user): a rule's result may stand for an unnormalised function, for
  whatever reason; the log scale is the scalar with `message = exp(logscale) · result`. Belief
  propagation is the common case where it is known, not its definition; a naive variational
  message has none. BayesBase (`compute_logscale`) and ExponentialFamily already treat it as core
  design, and the engine only propagates it where it can be computed.
- **It is part of the message**, not an annotation: `Message{D, L}`, `Marginal{D, L}`. Annotations
  stay the generic side channel for arbitrary information. v5 carried the log scale with the
  message too (its rules returned `(message, addons)`, `AddonLogScale`); this is that idea,
  declared and structured.
- **A rule declares it statically**, `logscale = …`: a constant (an `Irrational` such as
  `loghalf` keeps the float type; `-logtwo` would not, since negating an `Irrational` gives a
  `Float64`), a function of the body's slots, or `from_body`, the body returning
  `with_logscale(result, logscale)`. The declaration is rule metadata (`RuleSpec.logscale`), so
  which rules provide one is visible in the code and to tooling. Omitted, it is an
  `UndefinedLogScale(:no_declaration, spec)`.
- **Undefined propagates instead of erroring**: through products, with its first reason; only
  `require_logscale` (a consumer such as Mixture, or a user) errors. v6's zero for all-point-mass
  inputs is gone: it was wrong in general (Bernoulli towards `p` from an observed `out` is
  `log(1/2)`), and the rules it covered declare zero where that is exact.
- **Incoming log scales** are `args.logscale.m[:x]`, for a rule declaring `reads_logscale = true`
  (checked when the rule is resolved). `args` carries them in a third, dispatch-free type parameter
  of `RuleArgs`; the engine builds them whenever it tracks log scales, since it builds `args`
  before it knows the rule. They no longer go through `ann.m`.
- **`RuleResult`**: every public call of a rule returns one, one shape for every rule (v5's
  `@call_rule` returned a message or a tuple depending on an option). The engine never builds
  one; inlined, `getresult(message_passing_rule(...))` allocates nothing, as the routing gates
  show. Its generic getters other than `getresult`, `getlogscale`, `getrule` and `getannotations`
  are public, not exported, since RxInfer (and users) define `getcontext` and `getarguments` of
  their own. It must support **rich visualisation** (terminal, Jupyter, Pluto, Documenter, via
  `show(io, mime, x)`): recorded as a requirement, not built in this change.
- **The `product` service is removed**: the switch rule computes `compute_logscale` of the
  product itself, under the strategy on its algorithm, `MixtureBP(; prod = GenericProd())`.
- **The engine** tracks them with `logscales = true` (an activation option; RxInfer's `infer`
  keyword and option). Otherwise messages carry `nothing`, which absorbs in products, so there is
  no cost. Observations and constants carry zero; initial messages, fallback messages and form-
  constrained products are undefined.
- **Rejected**: keeping `product` as the variables' `MessageProductContext` (an engine type in
  rules, still one user); a custom service Mixture declares (the engine still has to supply it);
  scaled messages as values flowing into rules (every rule's dispatch would see the wrapper — the
  output-only `with_logscale` avoids it); dropping log scales.
- **Deferred, before 7.0**: deriving the log scales of Standard's remaining belief-propagation
  rules, each verified by quadrature, with a gate for them.

---

## 4. Corrections — read this before re-proposing anything

Claims the assistant made that were **wrong** and should not be revived:

1. **"Run the allocating rule once to discover the buffer."** Circular; if only the in-place
   rule exists there is nothing to run.
2. **"Derive in-place by rewriting the tail into `copyto!(out.μ, mexpr)`."** Defeats the
   purpose — `mexpr` is already allocated.
3. **"`rule!` allocates nothing in steady state."** In-place ≠ non-allocating.
4. **"Purity can be tested by running twice and hashing."** Tests determinism, not purity.
   (But auditing *declarations* at runtime is valid and was adopted.)
5. **"Per-rule Reactant compilation is a dead end."** A benchmark claim about small-`d` CPU,
   over-generalised into an architectural verdict. It is a research path.
6. **"Generation-direction name mangling is safe."** `q(a, b_c)` and `q(a_b, c)` collide.
7. **"No `prod!` exists."** `BayesBase` exports `prod!` already; it is merely unused.
7b. **"`rts_smoother` is unused."** That name does not exist; the function is `smoothRTS`
    and it *is* used by the delta unscented and linearization marginal rules. A bad grep.
7c. **"Approximation methods should be algorithms and depend on the base package."** They
    are utilities that algorithms use; the package stays standalone. See §3.9b.
7d. **"ReTestItems is needed for name/tag filtering."** TestItemRunner's filter already
    receives `(filename, name, tags)` — its own docstring example filters on tags. The
    change is ~10 lines in `runtests.jl`, not a package swap. Its only real advantage is
    distributed parallel workers, which is a CI wall-clock argument.
7e. **"`CVIProjection` needs its own package."** Over-engineering for a leaf component. Once
    the layout half stops being engine code, a weakdep extension of the Delta node package
    is enough. See §3.9b-ii.
7f. **"`cvi_setup!`/`cvi_update!` indicate the newer CVI work."** They belong to the **old**
    `ProdCVI`; `ReactiveMPOptimisersExt` predates `CVIProjection` by over a year.
8. **Threading needs an explicit schedule IR with layered fork-join.** Over-engineered for
   what was asked; purity is the actual requirement.
9. **A `Test` package extension for test tooling.** Would force test-only deps into the
   base's `Project.toml`. Separate package in `[extras]` instead.
10. **`@marginalrule` was never proposed for deletion** — but the question was asked badly
    enough to cause confusion. To be unambiguous: **`@marginalrule` stays** (106 definitions,
    called for every multi-variable cluster; deleting it would remove structured VMP).
    `Marginalisation`/`MomentMatching` — the dead `vconstraint` label — is what goes.
11. **"The design needs `ScopedValues`, so the Julia floor must rise to 1.11."** It does
    not. The context is an ordinary object passed into the rules; a default argument gives
    the same behaviour on 1.10. One sentence of a draft was mistaken for a constraint.
12. ~~**"`CompanionMatrix` must be used by the autoregressive node."** It is not — AR has its
    own `ARTransitionMatrix`. Plausible from the name and the concept, false in the code.~~
    **This correction was itself wrong** (Phase 6 entry brief): the user's claim was right. AR
    builds a `CompanionMatrix` with `as_companion_matrix` at seven sites; `ARTransitionMatrix`
    is its noise covariance. The search behind the correction looked for the type's name.
13. **"`Optim` is used by `ContinuousTransition`."** No: that grep matched the words
    "Optimized" and "Optimizer". Only `laplace.jl` uses the package.
14. **"The `:` in `NormalMeanVariance(:out)` is inconsistent decoration, so drop it."** The
    observation was right and the conclusion backwards. Once the body is a real lambda over a
    real object, `args.m[μ]` is an `UndefVarError` and only `args.m[:μ]` works, so the colon
    becomes load-bearing and spreads *into* the declarations rather than out of them. See
    §3.14.
15. **"`@logscale` becomes `annotate!(ctx, :logscale, v)`."** No. `ctx` is immutable
    infrastructure the rule *reads*; annotations are a mutable sink the rule *writes*, and
    they are separate body slots. The mistake came from `PLAN.md` itself, whose context
    bullet listed "annotations" among the context contents — now corrected there too.
16. **"Storing the rule body inside the `RuleSpec` forces a dynamic call."** Half right, and
    the half that is right was measured badly. It forces one **only where the compiler cannot
    see which body is in the field** — at a call site that can reach a single rule the whole
    spec is erased, whatever its representation. The early measurements that appeared to
    settle this were contaminated by constant folding: a spec constructed inline inside an
    inlinable `resolve` folds away entirely and reports zero allocations for *every*
    representation, including the one the table called bad. See §3.14 for the corrected
    three-way comparison and the adopted design (no type parameters).
17. **"`inplace` should be a type parameter."** No. Measured indistinguishable from a plain
    `Bool` field — zero allocations either way, including with the split body shapes where an
    allocating body takes `(args)` and an in-place body takes `(output, args)`. It adds no
    distinct spec types either, since the body type is already unique per rule. A flag hoisted
    into a signature is pure cost and a downstream-instability trap.

18. **"#11 needs a static capability table in the base registry."** Redundant. The host
    already guards this with the `is_delta_node_compatible` trait, specialised in the host
    with an error naming the package, and flipped by the extension. See §3.16.
19. **"Key a joint marginal by an internal `Symbol("y,x")`."** Collision-free, but it
    forms a symbol at run time, which is slow and was never the design. The key is the
    member tuple in the type, `Val((:y, :x))`. See §3.16.
20. **"Phase 4.5 needs a bridge so v6's engine can call the new rules."** It does not. The
    slice's hard cases run through engine code that is being deleted, and a live v6
    comparison is impossible anyway (same UUID), so the bridge would be adapters for dead
    code. The real engine is written directly, against recorded v6 fixtures. See §3.18.
21. **"`DeferredMessage` is a staleness trap; snapshot its inputs at emission."** It holds
    source observables and reads their latest values at materialisation, and that is
    load-bearing for the correctness of reactive message passing, not a defect. The new
    engine materialises exactly as v6 does. See §3.19.
22. **"RxInfer is being refactored too, so the engine can drop its surface and own a new
    construction API wholesale."** An over-reading. v7 keeps the reactive machinery and
    replaces rule lookup and invocation plus node and rule definition and creation. RxInfer
    adapts where that breaks it, and the adaptation is expected to be small. See §3.19.
23. **"`BP` and `VMP` are algorithms."** They are not; they are what the one default
    algorithm does under different factorisations. An algorithm selects rules: a rule
    switcher, or a node's own. Modelling a structured factorisation as a `Structured` algorithm,
    as a test once did, repeats the same mistake. See §3.20.
24. **"Moving to the new rule system needs a dual path, node by node."** It needs none; nothing
    outside the branch depends on the old one. The engine switches outright and unported
    nodes wait in `legacy/v6/`. See §3.22.
25. **"The per-module registry scopes rules; rules should be pushed into one central
    registry that lookup uses."** Lookup never reads a registry: it is the base package's
    method table, which is global already. The per-module registries are introspection data,
    per module only because of precompilation, and `registries()` joins them. Kept as it is
    (user). See §3.23.
26. **"The mixture's `reverse(...)` is observationally inert; only emission order is at
    stake."** Half right. Reversing the members within a group is inert. The order of the
    groups, precisions before means, is the VMP update schedule, and changing it changes the
    trajectory: the same optimum, reached in about 7 iterations instead of about 13. Measured
    in case (c). See §3.24.
27. **"Ported with v6's own tables and node tests" means ported correctly.** Not when v6's
    tests pin v6's errors. Phase 5 step 3 ported the Gamma and GammaInverse average energies
    with their v6 math (E[x]/E[θ] for E[x/θ], θ/E[x] for E[θ/x]), and the copied node-test
    values agreed, because they were computed with the same mistake. A review against the
    density caught it; both are corrected and declared (ReactiveMP.jl#672). A v6 expected value
    is evidence of what v6 does, not of what is right: hand-derive, or verify against the node
    definition, where the tooling allows.

---

## 5. Empirical results

Two open questions were checked against the real packages (commit `8aa85ea2`). Four more were
checked during the pre-Phase-0 audit (commit `d0f45cea`) and are recorded at the end.

**Aqua piracy.** `Aqua.Piracy.hunt(ReactiveMP)` reports exactly **3** pirate methods, none
of them rules: `default_prod_rule` and `prod` for `Uniform`×`Beta`
(`nodes/predefined/uniform.jl:6,9`) and a `dot` overload for `ForwardDiff.Dual`
(`fixes.jl:12`). The check can be enabled today.

**But it is vacuous for *message* rules, permanently.** *(Phase 4.5 found the rest of a rule
package is flagged — `nodespec`, `nodefunction`, average energies and marginal rules for
another package's distribution — and declares its node types owned with `treat_as_own`.)*
Aqua treats a `DataType` as foreign only if
the type *and every parameter* is foreign, and `is_foreign(::Symbol)` is unconditionally
`false`. Measured: `is_foreign(Val{:out}) == false`, `is_foreign(Val{1}) == true`. Every
rule target carries the edge name as a `Symbol` type parameter, so no rule can ever be
flagged, whoever defines it. Enable the check, but do not cite it as evidence the package
split is piracy-clean. **This also removed piracy as an argument for the ruleset axis.**

**Licensing.** `PolyaGammaHybridSamplers` is GPL-3 and is a direct `[deps]` entry;
ReactiveMP ships MIT. Real conflict, propagates to RxInfer, accepted and deferred (§3.9c).

**Approximation usage, updated after the CVI removal decision.** Surviving approximations
need no cubature package *(the list first given here — ForwardDiff, Distributions, Random,
LinearAlgebra — was wrong: the ported package is pure numerics on LinearAlgebra and
FastCholesky, with ForwardDiff arriving with Linearization, Phase 4.5 step 3)*;
`DiffResults` leaves with old CVI. `Optim` leaves
ReactiveMP entirely (only `laplace.jl` used it). `FastGaussQuadrature` follows `ghcubature`
to the Pólya package. `DomainIntegrals` and `HCubature` go to the test-utils package — they
are used by the rule-comparison quadrature at `src/rule.jl:1464,1540`, which is test
machinery. `DomainSets` stays with the standard rules (`normal_mean_variance/var.jl`; the ported
`gamma_shape_rate/a.jl` does not need it).

**BayesBase coverage.** The `ExponentialFamily` ban holds. Across every prospective base
file the only `ExponentialFamily` mentions are inside docstring examples. All 37 names the
base needs are available from BayesBase (36 exported; `kldivergence` defined but
unexported). Caveat: "thin" means thin in *direct* deps — BayesBase transitively brings
`Distributions`, `DomainSets`, `StaticArrays`, `StatsBase`, `StatsFuns`,
`SpecialFunctions`, `TinyHugeNumbers`. So `DomainSets` does not move out to the
approximations package as originally written.

### Checked during the pre-Phase-0 audit

**`RuleSpec` representation.** Three representations measured on Julia 1.13, at a call site
that can reach more than one rule, with resolution held inferable but not constant-foldable.
Results and the adopted design are in §3.14. Two things were established that no earlier
measurement had: `inplace` as a type parameter is worth nothing, and the previously published
reproduction could not distinguish the representations at all, because constant folding erased
the difference.

**Every generated rule method reports the same source location.** `@rule` splices its method
into a `quote` block in `src/rule.jl`, so all ~490 generated methods carry
`Method.file == "rule.jl"` and `Method.line == 372` (`marginalrule`: `:406`; `358`/`392` before the Runic reformat). Consequences:
an ambiguity report that cites `rule.jl:372` (or `:406` for `marginalrule`) is naming the *template*, not a rule — an earlier
version of `PHASES.md`'s ambiguity table read it as the arithmetic catch-alls, which actually
live in `src/rules/addition/in2.jl:1`, `src/rules/subtraction/{out,in1,in2}.jl:1` and
`src/rules/multiplication/marginals.jl:31`. And `Method.file`/`line` cannot identify a rule at
all today, which is why v6 recovers names from the *signature* instead
(`get_node_from_rule_method`, `src/rule.jl:1690-1716`). An independent argument for the
registry.

**The mixture `reverse(...)` is observationally inert.** It is not in `mixture.jl` — it is
`normal_mixture.jl:176,183` and `gamma_mixture.jl:166,173`, and it reorders only the
`combineLatest` *trigger* tuple while the emitted payload comes from `map_to` with the groups
un-reversed. `combineLatest` gates on the set of streams, not their order, so nothing
observable depends on it. Open item #6 is therefore a scheduling pin, not a numerical one.
*(Corrected in §3.24 and Correction 26: the trigger order is the subscription order, which is
the update schedule, and its group order changes the values.)*

**`Mixture`'s `RequireMarginal` path is unreachable, mechanically.** `mixture.jl:146` defines
`functional_dependencies` with three positional arguments; its only caller
`with_functional_dependencies` (`dependencies.jl:119-126`) passes four. Dispatch falls through
to the generic method, whose first statement is `getlocalclusters(factornode)` — and
`MixtureNode` has no such method, so the policy that `collect_functional_dependencies`
explicitly accepts `MethodError`s at activation time.

**`Message{D}`'s type parameter, and what actually costs time (2026-09-25, Phase 7; the user's
question, nothing decided).** Full record, variants, scripts and raw results:
`investigations/message-type-parameter/`. Measured at `851559a4`, through RxInfer's v7 branch; every
variant bit-identical to HEAD.
- **Dropping the parameter (`data::Any`) is slower everywhere:** per `infer` 1.13–1.65×, per VMP
  iteration 1.9–3.2×. Rule resolution becomes a runtime dispatch over every rule package (27 ns →
  0.7–1 µs), and every pairwise product a runtime `prod`; the streams, already abstract and
  allocation-free, gain nothing.
- **Function barriers recover most of it** (untyped + one-method kernels): per `infer` 0.96–1.12×
  of HEAD, but the graph loops stay 1.3× slower, each barrier's dispatch and allocation falling on
  every call and product.
- **Two costs in HEAD are the real target.** Building `Message{D}`/`Marginal{D}` from an
  `Any`-typed value costs 510 ns (`Core._compute_sparams`), on every rule output; building callback
  events nobody listens to costs about 270 ns per rule call. Fixing both while keeping `Message{D}`
  (a `@noinline` constructor at 4 sites, lazy `@invoke_callback` at 12) gives per `infer`
  0.76–0.96×, per iteration 0.37–0.75×, rule calls 0.14–0.25×, compile time unchanged.
- Typed streams are not needed for these gains. Next suspects (inferred): the abstract `RuleSpec`
  behind `execute_rule`, and the `Any` tuple a product returns. Input to the end-of-refactor
  performance pass.


---

## 6. Still open, and why

`PLAN.md` owns the stable open-item numbers; `PHASES.md` assigns their decision gates.
The original discussion left these questions:

- ~~**Rule syntax final details (#1)**~~ — **RESOLVED**, see §3.14. The outbound-edge
   spelling is `target = :out`, and `algorithm` is a top-level keyword on both the rule and
   the node. Ten representative rules are still written by hand in Phase 0, but now as
   validation of a decided form rather than as a way of choosing one.
- **`aligned` selector generality (#2)** — everything in-tree is `k ↔ k`. `q[:p][f(k)]`
   extends naturally; deliberately not built until something needs it. (Not `q[:p[f(k)]]`,
   which parses as `(:p)[f(k)]` — indexing a `Symbol`. See §3.14.)
- ~~**Per-(target, factorisation) group selection (#3)**~~ — **RESOLVED at the Phase 3
   sign-off: per target** (§3.16). Selection that genuinely varies with factorisation is a
   distinct algorithm, not a syntax axis. *(Refined in §3.20: only a node that ignores the
   factorisation needs its own algorithm.)*
- **The ruleset axis (#4)** — scoped rule tables (`Overlay(mine, standard)`). Introduced by the
   assistant, never requested. Its piracy argument is now dead for rules (see §5); `algorithm` may
   already cover the "controllable dispatch" goal. **DEFERRED in Phase 0**, and the
   rule-fallback contract it was holding up was specified independently there (§3.15).
- ~~**The engine step is under-planned.**~~ The rule layer is designed in detail;
   "rewrite ReactiveMP against the new base" hides the mixture `activate!` work, the
   dependency-to-stream wiring for variadic groups, and the `Message`/`DeferredMessage`
   envelope changes. Wants its own session. *(Resolved: the Phase 4.5 design brief was
   signed off, §3.19, and step 4 is planned in `PHASES.md`.)*

Open as of the Phase 4.5 reconciliation:

- ~~**The `Message` representation**~~ — **RESOLVED in step 4 by benchmark: `mutable` with
  `const` fields stays.** Through the equality chain it was about 10% faster and 40% lighter
  than an immutable struct; on a BP chain the immutable one was 6–8% faster but still 25%
  heavier. The numbers are in `PHASES.md` § Phase 4.5, step 4.
- **Typed annotations** (`Message{D, A}`, brief item 3) — not built in step 4, which kept the
  `AnnotationDict` so as to change one thing at a time; the retained-value test pins that
  nothing mutates it after materialisation. Revisit with the log-scale milestone (§3.23).
- ~~**Declared dependencies, groups and a declared free-energy partition in the engine**~~ —
  **RESOLVED in case (c)** (§3.24).
- **User rule sets beyond one-level extensions** — the registry stays introspection only
  (decided, §3.23). If extensions ever need to overlay a node's own algorithm, the recommended
  form is `AlgorithmExtension{Parent}`, not a registry axis. Folded into #4.
- ~~**Delta's own algorithm**~~ — **RESOLVED in case (d)**: `DeltaApproximation` and the
  engine's `getnodefn` (§3.25).
- ~~**Distributing a `FactorizedCluster`**~~ — **RESOLVED in Phase 5 step 2**: a joint input
  holding one reaches every rule as its blocks, message rules as well as the average energy,
  where v6 decomposed only for the average energy (`PHASES.md` § Phase 5).
- **A joint holding only some members of a group** with other interfaces — `activate!` refuses
  it; built when a node needs one.
- **`Uninformative` as BayesBase's product identity** — Standard's `UninformativeProd` has 0
  ambiguities only by writing out its overlaps with BayesBase's own rules; BayesBase owning the
  identity, as it does `missing`, removes them by construction. The Uniform(0, 1)×Beta product
  is the same kind of upstream item (Phase 5 step 3, user).
- **#11's two follow-ups** — `DeltaApproximation`'s positional constructor bypasses the
  compatibility guard, and the error names neither a package nor an alternative method;
  Phase 6, with `CVIProjection`.
- **Default initial messages** — how a node declares one for a rule that depends on its own
  edge (Probit), separately from `dependencies` (§3.21). Needed when Probit is ported.
- **Log scales** — preserved as v6 has them, gaps included; fixing them is a milestone of its
  own after the migration.
- ~~**The models package's name**~~ — *settled by §3.30: no such package; each node gets its own.*
- **The Julia floor** — 1.13 only until registration, when 1.10 support is reconsidered
  (§3.22).

Review added #9–#13 (all but #13 settled at the Phase 3 sign-off, §3.16): belief/entropy separation, buffer ownership, capability metadata,
context services and the numerical protocol. These block API freeze, not preparation or
the spike. #14 was the preparation inventory, resolved in Phase P. Purity/RNG contracts and derivative checks
are separate requirements. Reactant/StableCholesky (#5) remain a separate effort; mixture
regressions and edge identity (#6–#7) were engine integration requirements, resolved in
Phase 4.5 case (c) (§3.24; #7's RxInfer side is Phase 7).

---

## 7. Recommended order

Use `PHASES.md` as the authoritative checklist. Preparation comes first: baselines,
disposition inventory and pinned comparison environments. Then the throwaway spike checks
dispatch overhead, syntax, allocation, hard context services and delta execution semantics.
Finding a failure here should cost days, not months.

Then circulate `PLAN.md` + spike results for external feedback, *before* building the macro,
because the macro is where effort starts compounding.

In parallel with waiting: migrate ReactiveMP's tooling (Runic, Aqua, and name/tag filtering
in `runtests.jl` — **not** a runner swap). It is independent, low-risk, and compounds —
every later session runs faster.

Then base package → test utils with a bounded numerical oracle → **Phase 4.5: the engine
design session and the engine's first cut, refactored in place, as a clean cut** (§3.18–3.25; closed) →
bulk standard-rule
migration into it → approximations and node packages → completing the engine → the cleanup
of historical remarks (Phase C) → coordinated release. Start strict downstream CI as soon as
compatible development revisions exist, rather than waiting until release.

Rule kernels and test utilities can be developed independently of the engine, but that
does not prove the interface correct. Phase 4.5 must pass before bulk migration, and full
engine/downstream integration must pass before release.
