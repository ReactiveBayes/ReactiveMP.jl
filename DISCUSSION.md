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

Numbers: 48 `@node`, 390 `@rule`, 108 `@marginalrule`, 57 `@average_energy`, ~24k SLOC,
491 rule definitions total, 172 `@call_rule`/`@test_rules` sites in tests.

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
bolted-on mechanism into ordinary algorithms.

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
3. **`m[x]` / `q[x]` — adopted.** The decisive property is that **declaration and body use
   the same spelling**. There is no derived name, so there is nothing to collide. The
   binding problem does not get solved, it disappears.

Consequences that fell out for free:
- `m[μ]` and `q[μ]` in one signature is unremarkable — two separate containers. An earlier
  design (from a subagent) had invented a `Both{...}` wrapper type and a
  "key a cluster by its lexicographically smallest member" rule to handle this in one
  merged container; all of that evaporated.
- `q[y, x]` is a structural cluster (lowers to `getindex(q, Val((:y,:x)))`), so nothing is
  ever split on `_` and interface names may contain underscores again.
- `m[inputs...]` expresses a variadic group, replacing `ManyOf{N,T}` and its `where {N}`.

Also considered and rejected: a `where { q(a,b) <: T, m(x) <: S }` block (user's proposal).
It parses fine — verified with `Meta.parse`, including mixed type-parameters and slots in
one brace list — and `<:` is more *honest* than `::` (the macro already converts `::T` into
`Message{<:T}`). Rejected because `where` already means type parameters to every Julia
reader, and rules genuinely need the original meaning (delta rules carry
`where {N, M <: Unscented, L, I <: NTuple{L, Function}}`).

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
remain open in `PLAN.md` item #9 and must be settled before the API freezes.

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

**Landing point (user's `@allocate` idea):** the buffer shape must be *declared*, because
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

User's semantic for naming the buffer: in an `inplace` rule, `m[<target edge>]` **is** the
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
per rule. `BP` is pure so ~350 rules need no annotation; BIFM and, in the original design,
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
shapes, both need the output container declared up front. So `@allocate` +
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
`MessagePassingRulesApproximations` + `MessagePassingRulesTestUtils` + the `ReactiveMP`
engine. The approximation package's contents and layering changed below.

**Pivot (user):** test tooling is its own **package**, not part of the base. The assistant
had proposed a `Test` package extension; the user pointed out testing deps belong in
`[extras]`. The assistant's `weakdeps` remark was explaining why its *own* extension idea
was bad (an extension's deps must be weakdeps of its host, so cubature/Turing would land in
the base's `Project.toml`) but stated it confusingly. Outcome: separate package, listed
under `[extras]` by consumers, no weakdeps or extensions involved anywhere.

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
migration entries, including cases with no replacement (`PLAN.md` item #14).

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
user's framing is better: "how do I approximate this integral" is numerics; "which update
scheme am I running" is an algorithm. The delta algorithm *uses* Unscented; it is not
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
in the already-loaded host or a static table (`PLAN.md` item #11).

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
- Assistant's addition: **in-body macros need not be exported at all.** `@allocate` and
  `@logscale` only appear inside a rule body, so the enclosing definition macro can
  recognise and rewrite them. Short names, zero namespace footprint, and using one outside
  a rule body becomes a clean error.
- Invocation macros stay **short** — the "long but unambiguous" argument inverts for
  something typed constantly at a REPL in front of students.

### 3.11 Testing and documentation

Strict TDD, failing test first in PRs unless justified.

The leverage point: **because `@rule`/`@node` emit data, "every rule has a test" becomes a
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
**coverage matrix** (edges × algorithms, showing which rules exist) that serves students
and developers equally, `Base.show` MIME methods for REPL and notebooks, and visualisations
behind the extension mechanism following GraphPPL's pattern. Error messages count as
pedagogy — the near-miss display with per-slot diffs is a teaching tool.

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
7e. **"`CVIProjection` needs its own package."** Over-engineering for a leaf component. Once
    the layout half stops being engine code, a weakdep extension of the Delta node package
    is enough. See §3.9b-ii.
7f. **"`cvi_setup!`/`cvi_update!` indicate the newer CVI work."** They belong to the **old**
    `ProdCVI`; `ReactiveMPOptimisersExt` predates `CVIProjection` by over a year.
7d. **"ReTestItems is needed for name/tag filtering."** TestItemRunner's filter already
    receives `(filename, name, tags)` — its own docstring example filters on tags. The
    change is ~10 lines in `runtests.jl`, not a package swap. Its only real advantage is
    distributed parallel workers, which is a CI wall-clock argument.
8. **Threading needs an explicit schedule IR with layered fork-join.** Over-engineered for
   what was asked; purity is the actual requirement.
9. **A `Test` package extension for test tooling.** Would force test-only deps into the
   base's `Project.toml`. Separate package in `[extras]` instead.
10. **`@marginalrule` was never proposed for deletion** — but the question was asked badly
    enough to cause confusion. To be unambiguous: **`@marginalrule` stays** (108 definitions,
    called for every multi-variable cluster; deleting it would remove structured VMP).
    `Marginalisation`/`MomentMatching` — the dead `vconstraint` label — is what goes.

---

## 5. Empirical results

Two open questions were checked against the real packages (commit `8aa85ea2`).

**Aqua piracy.** `Aqua.Piracy.hunt(ReactiveMP)` reports exactly **3** pirate methods, none
of them rules: `default_prod_rule` and `prod` for `Uniform`×`Beta`
(`nodes/predefined/uniform.jl:6,9`) and a `dot` overload for `ForwardDiff.Dual`
(`fixes.jl:12`). The check can be enabled today.

**But it is vacuous for rules, permanently.** Aqua treats a `DataType` as foreign only if
the type *and every parameter* is foreign, and `is_foreign(::Symbol)` is unconditionally
`false`. Measured: `is_foreign(Val{:out}) == false`, `is_foreign(Val{1}) == true`. Every
rule target carries the edge name as a `Symbol` type parameter, so no rule can ever be
flagged, whoever defines it. Enable the check, but do not cite it as evidence the package
split is piracy-clean. **This also removed piracy as an argument for the ruleset axis.**

**Licensing.** `PolyaGammaHybridSamplers` is GPL-3 and is a direct `[deps]` entry;
ReactiveMP ships MIT. Real conflict, propagates to RxInfer, accepted and deferred (§3.9c).

**Approximation usage, updated after the CVI removal decision.** Surviving approximations
need `ForwardDiff`, `Distributions`, `Random`, `LinearAlgebra` — **no cubature package**;
`DiffResults` leaves with old CVI. `Optim` leaves
ReactiveMP entirely (only `laplace.jl` used it). `FastGaussQuadrature` follows `ghcubature`
to the Pólya package. `DomainIntegrals` and `HCubature` go to the test-utils package — they
are used by the rule-comparison quadrature at `src/rule.jl:1438,1514`, which is test
machinery. `DomainSets` stays with the standard rules (`normal_mean_variance/var.jl`,
`gamma_shape_rate/a.jl`).

**BayesBase coverage.** The `ExponentialFamily` ban holds. Across every prospective base
file the only `ExponentialFamily` mentions are inside docstring examples. All 37 names the
base needs are available from BayesBase (36 exported; `kldivergence` defined but
unexported). Caveat: "thin" means thin in *direct* deps — BayesBase transitively brings
`Distributions`, `DomainSets`, `StaticArrays`, `StatsBase`, `StatsFuns`,
`SpecialFunctions`, `TinyHugeNumbers`. So `DomainSets` does not move out to the
approximations package as originally written.

---

## 6. Still open, and why

`PLAN.md` owns the stable open-item numbers; `PHASES.md` assigns their decision gates.
The original discussion left these questions:

- **Rule syntax final details (#1)** — `m[x]`/`q[x]` settled in principle; the outbound-edge
   spelling and where `algorithm` sits in the header are not. Best resolved by hand-writing
   ten representative rules and reading them.
- **`aligned` selector generality (#2)** — everything in-tree is `k ↔ k`. `q[p[f(k)]]` extends
   naturally; deliberately not built until something needs it.
- **Per-(target, factorisation) group selection (#3)** — assumed per-target. Works for all four
   in-tree cases because the mixtures pin their factorisation. A one-way door in the syntax.
- **The ruleset axis (#4)** — scoped rule tables (`Overlay(mine, standard)`). Introduced by the
   assistant, never requested. Its piracy argument is now dead (see §5); `algorithm` may
   already cover the "controllable dispatch" goal. May remain deferred; existing rule-fallback
   semantics still need a contract independently of this choice.
- **The engine step is under-planned.** The rule layer is designed in detail;
   "rewrite ReactiveMP against the new base" hides the mixture `activate!` work, the
   dependency-to-stream wiring for variadic groups, and the `Message`/`DeferredMessage`
   envelope changes. Wants its own session.

Review added #9–#13: belief/entropy separation, buffer ownership, capability metadata,
context services and the numerical protocol. These block API freeze, not preparation or
the spike. #14 is the preparation inventory. Purity/RNG contracts and derivative checks
are separate requirements. Reactant/StableCholesky (#5) remain a separate effort; mixture
regressions and edge identity (#6–#7) are engine integration requirements.

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

Then base package → test utils with a bounded numerical oracle → **Phase 4.5 engine
integration slice** → bulk standard-rule migration → approximations and node packages →
full engine integration → coordinated release. Start strict downstream CI as soon as
compatible development revisions exist, rather than waiting until release.

Rule kernels and test utilities can be developed independently of the engine, but that
does not prove the interface correct. Phase 4.5 must pass before bulk migration, and full
engine/downstream integration must pass before release.
