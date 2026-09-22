# Ground-up rewrite of the ReactiveMP rule/node system

## Context

ReactiveMP.jl is currently "a can of everything": one package holding the message-passing
engine, 48 node definitions, 390 `@rule`s, 108 `@marginalrule`s and 57 `@average_energy`s
(~24k SLOC). The rule/node layer is the oldest part of the codebase and it shows.

Goals: extract the rule/node layer into standalone packages so `AutoregressiveNode`,
`RxGP` etc. can exist independently of the engine; and, at the same time, redesign that
layer from the ground up rather than porting it. Clean break, new major version of
everything, no backwards compatibility, Julia floor may rise to 1.11+.

Verified problems driving the redesign:

- **A 10-argument god-function.** `rule(fform, on, vconstraint, mnames, messages, qnames,
  marginals, meta, annotations, node)` encodes one name→value mapping across *four*
  parallel arguments (`Val{(:μ,)}` + `Tuple{Message{<:T}}`, twice).
- **A dead dispatch axis.** `vconstraint` is hardcoded `Marginalisation()` at all 5
  construction sites; `MomentMatching` is dispatched on nowhere. 447 lines of boilerplate.
- **String mangling in the parsing direction.** `m_`/`q_` prefixes are stripped and joint
  marginals (`q_y_x`) are recovered by splitting on `_`. This is *why* `@node` forbids
  underscores in interface names. Generation-direction mangling doesn't fix it either:
  `q(a, b_c)` and `q(a_b, c)` both yield `q_a_b_c`.
- **Fragile introspection.** `showerror(::RuleMethodError)` decodes rules via
  `Base.arg_decl_parts` at fixed positional string offsets (`decls[2..9]`).
- **Compile-order coupling.** `@rule` queries the node registry at *macro expansion time*
  (`nodesymbol_to_nodefform`), so `@node` must be evaluated before any `@rule` for it.
- **Three parallel implementations** of one dispatch pattern: `@rule`, `@marginalrule`
  (own error type, own `showerror`, own test macro, no annotations) and `@average_energy`
  (separate 5-arg `score` function, own error path).
- **Rules are not pure.** `BIFMMeta` is mutated from inside message rules (`setH!`,
  `setΛz!`, `setμu!`, `setΣu!`); CVI stores `ForwardDiff` scratch in its meta; RxGP's meta
  holds a `counter` and a `Dict` cache. This blocks multithreading outright.
- **No in-place anything.** No preallocation, and `BayesBase.prod!` is exported but never
  used. Per message, steady state:
  `DeferredMessage` + `AnnotationDict` + `Message` + the rule's result; per degree-`d`
  variable, ~`2(d-1)` more.
- **Missing concept: variadic interface groups.** `Mixture` escapes `@node` entirely and
  hand-writes `interfaces`, `alias_interface`, `is_predefined_node`, `sdtype`,
  `collect_factorisation`, its node struct, `factornode`, `interfaceindex` and `activate!`.
  `GammaMixture` is a ~250-line clone of `NormalMixture`.
- **One engine leak in the whole rules tree.** `src/rules/mixture/switch.jl` allocates a
  throwaway `randomvar` inside a rule to reach product-with-log-scale machinery.

## Architecture

### Dispatch axes

Rules dispatch on: **node**, **target**, **algorithm**, **inputs**, plus a non-dispatching
**context**.

- **`algorithm` replaces `meta`.** `meta` today is already two things fused: an algorithm
  selector (`Unscented` vs `Linearization` vs `CVI`) and a parameter bag (AR order,
  kernels). Naming it `algorithm`, giving it a per-node default and making it swappable
  absorbs `Marginalisation`/`MomentMatching` — that axis disappears rather than being
  deleted separately. It also absorbs `RequireMessage`/`RequireMarginal`/
  `RequireEverythingFunctionalDependencies` (see Dependencies).
- **`context` is infrastructure, never dispatched on**: linear-algebra strategy (replacing
  48 global `cholinv` calls; `StableCholesky.jl` supplies strategies + workspace), RNG,
  output buffers, annotations, engine services. Explicit argument, never shared across
  tasks. A `ScopedValue` supplies the default at the engine boundary only.
- Rules declare which context **services** they need (e.g. `context = (:linalg, :product)`)
  so the engine can check availability and diagnose missing services.

### Rule surface

Role lives in a container, not a name prefix. `m[...]` and `q[...]` are two containers;
**declaration and body use the same spelling**, so no name is ever derived and nothing is
ever split. Interface names may contain `_` again.

```julia
@rule NormalMeanVariance(:out) (m[μ]::PointMass, m[v]::PointMass) = ...

@rule Mixture(:switch, algorithm = BP, pure = false) (
    m[out]::Any, m[inputs...]::Any
) = ...
```

- `m[μ]` and `q[μ]` in one signature is unremarkable — different containers. No wrapper
  type, no key-collision rule.
- `q[y, x]` is a structural cluster; lowers to `getindex(q, Val((:y, :x)))`.
- `m[inputs...]` is a variadic group, replacing `ManyOf{N,T}` plus its `where {N}`.
- Options are keywords (`algorithm`, `pure`, `inplace`, `context`), the signature stays
  positional. Growth room without taxing the ~350 rules that use none of it.
- `@marginalrule` and `@average_energy` keep their names and their distinct return
  contracts. They lower onto **three separate generic functions** (see Naming) that share
  one registry, one error path, one ambiguity checker and one test macro — shared
  *infrastructure*, not a single dispatch function. `@average_energy` gains the `algorithm`
  axis.

`@node` goes keyword-based, since it has the most to grow into (aliases are already
kwargs-in-disguise today):

```julia
@node Mixture(type = Stochastic, interfaces = [out, switch, inputs...])
```

### Naming

Everything is renamed; nothing inherits a generic name that downstream packages could
collide with.

**Generic functions — renamed and not exported**, since downstream packages add methods to
them: `rule` → `message_passing_rule`, `marginalrule` → `message_passing_marginalrule`,
`score`/average energy → `message_passing_average_energy`. In-place variants take `!`.
Reached qualified.

**Definition macros — renamed and exported**, long enough to be unambiguous since they are
typed once per definition: `@define_message_update_rule`, `@define_marginal_update_rule`,
`@define_factor_node`, `@define_average_energy`.

**In-body macros — short and *not* exported.** `@allocate` and `@logscale` only ever appear
inside a rule body, so the enclosing definition macro recognises and rewrites them during
its own expansion. They keep short names with zero namespace footprint (`@allocate` is
otherwise extremely collision-prone), and using one outside a rule body becomes a clean
error rather than a confusing one.

**Invocation macros — short, exported.** `@call_rule`/`@call_marginalrule` are typed
constantly in interactive and teaching use, so the "long but unambiguous" argument that
applies to once-per-definition macros inverts here. See Educational and interactive use.

### Educational and interactive use

A first-class goal, not a by-product. The system is taught with in the BMLIP course at
TU/e, where invoking rules by hand is a good way to show what message passing actually
does. The registry is what makes all of this cheap — today's `print_rules_table()` scrapes
`methods()` through `arg_decl_parts` string offsets, which is why nothing better was ever
built on it.

- **Manual rule invocation stays first-class.** `@call_rule`/`@call_marginalrule` keep
  working and keep **short names** — unlike the definition macros they are typed constantly
  at a REPL, so the `@define_*` verbosity argument runs the other way here. Consider a
  function-style entry point alongside the macro for interactive use.
- **Querying the registry**: `rules(NormalMeanVariance)`, `rules(NormalMeanVariance, :out)`,
  filtering by algorithm, and `@which_rule` (which `RuleSpec` fires for these inputs, with
  source location). "Does this node support VMP or only structured?" becomes a query,
  because the algorithm axis and the dependency spec are now data.
- **Rule coverage matrix** — edges × algorithms for a node, cells showing which rules exist.
  Simultaneously a teaching artifact ("when can I use VMP here?") and a development one
  ("what is missing in the port?").
- **Rich display via `Base.show` MIME methods** — `text/plain` for the REPL, `text/html` for
  notebooks, since the course uses them. Zero dependencies, works everywhere.
- **Visualisation via the extension mechanism**, following GraphPPL's pattern: diagrams of a
  node's interfaces/groups, of what a rule receives under a given factorisation, and of the
  coverage matrix activate when a plotting package is loaded, and otherwise fail with a
  descriptive "load X to enable this" message rather than a `MethodError`.
- **Error messages are pedagogy.** The "no rule found" output showing near-miss rules with
  per-slot ✓/✗ diffs is a teaching tool for students who get an input wrong, which is the
  common case in a classroom.

### Registry, errors, introspection

`@rule`/`@node` emit **data** (a `RuleSpec`/`NodeSpec`) alongside the method, stored as a
per-module `const` and discovered by scanning loaded modules. Do **not** `push!` into a
ReactiveMP-owned global — a downstream package's top-level `push!` runs during *its*
precompile and lands only in its own image.

This replaces `arg_decl_parts` string decoding entirely: "no rule found" has the target,
inputs and algorithm as live values. Distinguish the two failure classes that are
currently indistinguishable — *no rule of this shape* (wrong dependency/factorisation)
vs *type mismatch* (rule exists, arguments don't fit). Add `check_rules()` (validates
specs against node specs, and dependency specs against rule signatures) and
`check_rule_ambiguities()` (group by input-name set, `typeintersect` within groups only).

Removing the macro-expansion-time registry query means `@node` and `@rule` may appear in
any order, in any package, including weak-dep extensions.

### Dependencies as a language

Dependencies become a **property of the algorithm**, not the node; a node declares a
default algorithm. Written in the same `m[]`/`q[]` vocabulary as rules:

```julia
@node NormalMixture(
    type = Stochastic,
    interfaces = [out, switch, m..., p...],
    dependencies = [
        m[k] => (q[out], q[switch], q[p[k]]),
        p[k] => (q[out], q[switch], q[m[k]]),
    ]
)
```

Two axes separate cleanly:

- **Role** (message vs marginal) stays derived from the factorisation, as today.
- **Group selection** is the new declarative axis, and the only thing needing declaration.

A group selector maps the target index to a tuple of source indices. The four observed
modes are four values of one type; a user lambda is a fifth:

| selector | indices | arity | used by |
|---|---|---|---|
| `all` | `1:N` | `N` | `Mixture(:out)` |
| `allbutself` | `1:N \ {k}` | `N-1` | `DeltaFn((:in,k))` (delta's `TupleTools.deleteat`) |
| `aligned` | `{k}` | `1` | `NormalMixture((:m,k))` needs `p[k]` only |
| `none` | `{}` | `0` | `Mixture((:inputs,k))` |

**Hard constraint: selectors must have statically known output arity.** Otherwise the
`ManyOf` length is runtime-dependent, `ManyOf{N,T}` can't specialise, and dispatch
destabilises downstream. A boundary-dependent lambda like `k -> (k-1,)` violates this at
`k=1`; supported resolutions are a total selector (wrap/clamp/pad) or a distinct target
type for the boundary. Length-unions are unsupported.

The default (same cluster → messages minus self; other clusters → marginals) already
yields `allbutself` and `all` for free, so only narrowing needs declaring — `Mixture` needs
one line, `DeltaFn` none. The default fails by *over*-supplying, which surfaces as "no rule
matches" rather than a wrong answer.

For custom algorithms, dependencies stand alone and imply the factorisation rather than
deriving from it — requesting `q[a,b]` *is* declaring a and b share a cluster.
**Well-formedness condition:** marginals requested across all targets of a node must form a
consistent partition, or the node's BFE entropy term is undefined; an algorithm that
violates it deliberately must supply its own `score`.

### Purity

`pure` is declared on the **algorithm**, inherited by its rules, overridable per rule.
`BP` is pure, so the ~350 already-pure rules need no annotation; `BIFM` and CVI carry the
marker once on the algorithm rather than once per rule. Narrowing (pure rule under impure
algorithm) is always safe; **widening must taint the graph-level check**, or the
inheritance is unsound in the direction that matters. Multithreading is not implemented —
the flag exists so it can be, and so impure rules are diagnosable.

**Purity cannot be *proved* by testing** — running a rule twice and comparing outputs tests
determinism, and absence of side effects is not establishable by test. Writing a pure rule
is the author's declared responsibility.

**But the declarations can be audited at run time, and that is what matters in practice.**
See Engine diagnostics: `check_everything_pure` walks the graph and errors naming any rule
whose algorithm (or own override) declares impurity. Motivating case: differentiating
through inference with ForwardDiff (as several RxInferExamples do) requires every rule on
the path to be pure, and today someone getting wrong gradients has no way to find the
culprit.

Optionally, when that flag is on, a *mutation detector* can also run — `deepcopy` the
algorithm, run, compare with `==` — catching the in-tree offender (BIFM mutating its meta).
It only sees state reachable from that object, so it is a debug aid on top of the audit,
not a proof.

### In-place rules

**`inplace` means the result is written into a provided container. It does *not* mean
non-allocating** — intermediates may still allocate. Non-allocating is a stricter,
separate property with its own opt-in test flag.

`rule`/`rule!` mirroring `rand`/`rand!`. Because the allocating form can't be run to
discover a buffer (circular — if only the in-place rule exists there is nothing to run),
and because rewriting a body to `copyto!(out.μ, mexpr)` defeats the purpose (`mexpr`
already allocated), the shape must be declared:

```julia
@rule NormalMixture(:out, inplace = true) (
    m[out]::MvNormalMeanPrecision, q[m...]::Any, q[p...]::Any
) = begin
    @allocate MvNormalMeanPrecision(buffer_like(...), buffer_like(...))
    mul!(m[out].Λ, ...)   # writes straight into the buffer, no temporaries
end
```

- `@allocate` lowers to `allocate_result(rule, inputs, ctx)`; the body lowers to
  `rule!(buffer, inputs, ctx)`; `rule(args...) = rule!(allocate_result(args...), args...)`.
- In an `inplace` rule, `m[<target edge>]` **is the output buffer**. Such a rule may not
  also require the inbound message on that edge. This pins the output type in the
  signature, making `@allocate` type-checkable and the buffer type statically known.
- `buffer_like(x)` — allocation primitives that **dispatch on the source** (`Vector`→
  `Vector`, `SVector`→`MVector`, `ConcreteRArray`→on-device, …) rather than allocating a
  default and converting, which would repeat the same mistake one level up. Lives in and is
  documented by `MessagePassingRulesBase`, and is **the** extension seam for array types and
  devices — see Reactant below.
- In-place requires writable output fields. `Diagonal`, `SMatrix`, `PDMat`-backed
  (`Wishart` caches a Cholesky), `PointMass`, `ProductOf` are not; a trait gates it and the
  engine falls back to allocating. ~30 entries in BayesBase covers the useful set.
- Default is allocate; reuse is opt-in; a checked mode poisons recycled buffers so escapes
  fail loudly in CI. Equality-chain caches hold messages across iterations, so edges
  feeding one need a buffer per cache slot or exclusion from reuse.

Separately and with no ownership reasoning required: make `Message` immutable (all four
fields are already `const`; re-benchmark the comment claiming mutable is faster — it may
have been measured on `DeferredMessage`, which does need it), make annotations a type
parameter with a zero-field `NoAnnotations` default, and fold products over raw
distributions rather than `Message`s. That removes 2–3 of the 4 fixed allocations per
message and the `2(d-1)` per variable.

### Engine diagnostics

A family of opt-in audit flags on the inference engine, **all `false` by default**, all
reporting the *specific offending rule* rather than failing vaguely. Cheap, because the
registry already knows every declaration.

- **`check_everything_pure`** — errors if any rule in the graph declares impurity. For
  differentiating through inference (ForwardDiff), and as the prerequisite audit before
  multithreading is ever switched on. Optionally also runs the mutation detector (see
  Purity).
- **`check_everything_inplace`** — reports rules with no in-place implementation. A coverage
  audit for latency-sensitive use (robotics, real-time), not a correctness one.
- **Checked buffers** — poisons recycled output buffers so any escape fails loudly rather
  than silently corrupting. Run the full suite under this as a separate CI job.

These are also how a user *finds* what to fix: the answer to "why is my model allocating /
why are my gradients wrong / why can't I thread this" should be a rule name, not a hunt.

### Reactant — out of scope here, but the seam is not

Reactant itself and the `StableCholesky` integration are handled separately, **not in this
plan.** What this plan must not foreclose is the seam, which costs nothing now:

`buffer_like` dispatching on the source array type *is* the device seam — one rule body
runs on `Vector`, on `SVector`, and on device arrays with no rule-side change. And in-place
discipline and traceability discipline are nearly the same discipline: both forbid
materialising intermediates, both need static shapes, both need the output container
declared up front rather than discovered from a return value. So `@allocate` +
destination-passing is already the right shape, and `@allocate` is where device placement
would later belong. Keep `buffer_like` extensible and documented; build nothing else.

### Package split

The boundary is already almost clean: rules need only `getdata` and distribution math — no
Rocket, no factor graph, no scheduler. The exceptions are `src/rules/mixture/switch.jl`
(one `randomvar` call, becomes a context service) and 12 rules using `getnode`/`getnodefn`.
**Granularity beyond the packages below — splitting rules by distribution family — is
explicitly deferred.**

Because algorithms are now first-class *values*, the heavy numeric dependencies follow the
algorithms out of the core. Sorting today's 28 deps:

| package | contents | deps |
|---|---|---|
| `MessagePassingRulesBase` | macros, `Message`/`Marginal`, targets, algorithms, context, registry, dependency language, `buffer_like` | `MacroTools`, `TupleTools`, `BayesBase`, `LinearAlgebra` — **and nothing else** |
| `StandardMessagePassingRules` | standard distribution nodes + arithmetic (`+`, `-`, `*`, dot) | `ExponentialFamily`, `Distributions`, `StatsFuns`, `SpecialFunctions`, `FastCholesky`, `TinyHugeNumbers`, … |
| `MessagePassingRulesApproximations` | numerical utilities: `Unscented`, `Linearization`, `CVI`, CVI projection, optimizers, `smoothRTS`. **Standalone — does *not* depend on the base package** | `ForwardDiff`, `DiffResults`, `Distributions`, `Random`, `LinearAlgebra` |
| `MessagePassingRulesTestUtils` | all test tooling (see Testing) | quadrature / sampling, whatever verification needs |
| `ReactiveMP` | engine | `Rocket`, `UUIDs` |

The base package is genuinely thin. Two current deps are single-node-specific and should
follow their nodes out: `Tullio` (only `DiscreteTransition`) and `PolyaGammaHybridSamplers`
(only the Pólya nodes).

### Approximations are utilities, not algorithms

`MessagePassingRulesApproximations` is **not** part of the algorithm hierarchy and does not
depend on `MessagePassingRulesBase`. They are siblings. "How do I approximate this integral"
is a numerical utility; "which message update scheme am I running" is an algorithm. The
delta node's algorithm *uses* `Unscented`; it is not `Unscented`. Keeping them separate
leaves the numerics usable outside this ecosystem and keeps the dependency graph flat.

Node packages (Delta, Flow) depend on both.

**Measured disposition of `src/approximations/` (~1922 lines).** Deleting the unused parts
is what actually removes the heavy cubature dependencies — repackaging them would not have.

| file / symbol | fate | evidence |
|---|---|---|
| `unscented.jl`, `linearization.jl`, `approximations.jl`, `shared.jl` | → `MessagePassingRulesApproximations` | used by delta + flow |
| `rts.jl` (`smoothRTS`) | → same package | `rules/delta/unscented/marginals.jl:25`, `rules/delta/linearization/marginals.jl:27` |
| `cvi.jl` (`ProdCVI`, aliased `CVI`), `optimizers*` | **delete** | superseded — its own docstring reads *"`ProdCVI` is deprecated in favor of `CVIProjection`"* |
| `cvi_projection.jl` (`CVIProjection`, sampling strategies) | → Delta node package, implementation in its extension | see *CVI projection* below |
| `gausshermite.jl` (`ghcubature`) | → Pólya node package | only user is `multinomial_polya` |
| `sphericalradial.jl` (`srcubature`) | **delete** | no consumer |
| `gausslaguerre.jl` (`glcubature`) | **delete** | no consumer |
| `importance.jl` | **delete** | no consumer |
| `laplace.jl` | **delete** | no consumer |

Dependency consequences: **`Optim` leaves ReactiveMP entirely** (only `laplace.jl` used it);
**`DiffResults` leaves with `cvi.jl`**, its only user; **`ReactiveMPOptimisersExt` and the
`Optimisers` weakdep are deleted outright** — that extension exists solely to supply
`cvi_setup!`/`cvi_update!` for the removed method. `FastGaussQuadrature` follows
`ghcubature` to the Pólya package; `DomainIntegrals` and `HCubature` go to
`MessagePassingRulesTestUtils` (they serve the rule-comparison quadrature in
`src/rule.jl:1438`, which is test machinery); `DomainSets` stays with
`StandardMessagePassingRules` (`normal_mean_variance/var.jl`, `gamma_shape_rate/a.jl`).

What survives is small: `Unscented`, `Linearization`, `smoothRTS` and the shared
point/weight machinery, needing only `ForwardDiff`, `Random`, `LinearAlgebra` and
`Distributions` — **no cubature package at all**.

**This is the only capability regression in the whole plan — accepted, with conditions.**
Everything else here is API churn: renamed macros, new syntax, relocated code. Migration is
work, but no model loses a capability once the spelling is fixed. This one is different: the
delta node's built-in method set (`is_delta_node_compatible`) shrinks from
`{Unscented, Linearization, ProdCVI}` to `{Unscented, Linearization}`, so a model that runs
today on a plain `add ReactiveMP` may afterwards need a second package installed before it
runs at all. Note the two survivors are the same *kind* of method — moment propagation
through a deterministic function — while `ProdCVI` was the sampling-and-gradient one reached
for when linearization is not good enough. The capability is not lost (`CVIProjection`
supersedes it) but it moves behind an install.

**Decision: accept it, and make the diagnostic carry the weight.**

1. **Document it as a breaking change in the release notes**, called out explicitly rather
   than folded into the general list of renames — it is the one entry that requires an
   install, not an edit.
2. **The error must be actionable.** Using a delta node with a non-conjugate factor, or
   naming `CVIProjection` without the package loaded, must produce a message that names the
   package to install *and* the method to switch to. A `MethodError`, or a generic "no rule
   found", is a failure of this requirement. The registry has the information to do this
   properly (see § Registry, errors, introspection) — this is a concrete first customer for
   it, and a good test of whether those error messages are actually as good as claimed.

### CVI projection, and a hypothesis about delta layouts

`CVIProjection` spans awkward territory today: the type lives in `src/approximations/`, the
rules and a *layout* live in `ReactiveMPProjectionExt`, and the layout is engine code — it
constructs `MessageMapping`, calls `connect!`, wires Rocket streams.

**Hypothesis: `AbstractDeltaNodeDependenciesLayout` is a bespoke version of the dependency
language.** Three layouts exist (default, CVI, CVI-projection), each implementing
`deltafn_apply_layout` for the same four targets — `q_out`, `q_ins`, `m_out`, `m_in_k`.
That is ~684 lines whose genuinely distinct content is **twelve dependency declarations**;
the rest is identical wiring boilerplate. The CVI-projection layout's own docstring reads as
a dependency spec: *"`m_in_k`: uses the inbound message on the `in_k` edge and `q_ins`"*.

*Confidence: moderate.* This is inferred from docstrings and method shape, and the layout
does make ~10 real engine calls — some may be genuine topology rather than dependency
choice (`q_out` "mirrors the posterior marginal" sounds like stream aliasing, not a rule).
**Test it in Phase 0**, where the delta layouts are the hardest thing the dependency
language would have to express. Finding its limits on the hard case is the point.

**If the hypothesis holds**, the plan is: collapse layouts into dependency declarations
first, after which `CVIProjection` has no engine half at all — just an algorithm struct, a
dependency declaration, and rules. It then ships as a **weakdep extension of the Delta node
package**, keeping today's pattern, and no separate package is needed. A standalone package
stays the cheap upgrade later if anyone needs to depend on the projection rules, if compat
coupling to `ExponentialFamilyProjection` starts forcing awkward releases, or if it grows
its own CI. Promoting an extension to a package is far easier than the reverse.

**If it does not hold**, the layout half cannot live in a Delta-node extension — engine
internals belong to ReactiveMP and node packages must not depend on the engine — and a
separate package becomes necessary.

**API: keep it, do not redesign.** The functional surface carries over as-is —
`approximate_meancov`, `approximate_kernel_expectation`, `getpoints`/`getweights`,
`unscented_statistics`, `smoothRTS` — with `ctx` threaded in where `cholinv` is currently
called globally. Prettifying only; a redesign is explicitly out of scope for now.

**Deletions take their tests with them** (`test/approximations/{importance,laplace}_tests.jl`,
parts of `getpoints_tests.jl`/`shared_tests.jl`). Skim them first — a test may be the only
record of intended behaviour if anyone ever wants these back.

### Licensing — known, accepted, deferred

`PolyaGammaHybridSamplers` is **GPL-3** and is a plain `[deps]` entry, while ReactiveMP
ships an MIT `LICENSE`. Two nodes use it: `multinomial_polya.jl` and
`rules/binomial_polya/beta.jl`. This is a real conflict on `main` today and it propagates
downstream to RxInfer.

**Decision: live with it.** It has been the situation for over a year; this rewrite resolves
it rather than a separate fix. The resolution is a side effect of the split — the Pólya
nodes move to their own package, which may be GPL-3, and ReactiveMP goes back to being
honestly MIT.

**Hard constraint: `MessagePassingRulesBase` must not depend on `ExponentialFamily`.**
BayesBase exists precisely to hold this machinery. Where a needed piece is missing from
BayesBase, **add it to BayesBase**; do not reach for `ExponentialFamily`. Enforce with a CI
assertion that `ExponentialFamily` is absent from the base's dependency closure, so the
constraint survives contributors and sessions rather than eroding the first time someone
wants one convenience function.

**Measured — the constraint holds today.** Across every prospective base file
(`message.jl`, `marginal.jl`, `rule.jl`, `nodes/nodes.jl`, `annotations.jl`,
`constraints/form.jl`, `score/score.jl`, `helpers/*`, `variable.jl`) the only
`ExponentialFamily` mentions are inside *docstring examples*, not code. And all 37 names
the base needs are available from BayesBase — 36 exported, `kldivergence` defined but
unexported (`BayesBase.kldivergence`). Nothing is missing: the full `Message`/`Marginal`
statistics proxy list, `pdf`/`logpdf`, `paramfloattype`/`convert_paramfloattype`,
`PointMass`, `prod`/`prod!`, `GenericProd`/`ClosedProd`/`PreserveTypeProd`/`ProductOf`,
`default_prod_rule`, `deep_eltype`.

**Caveat on "thin": that is thin in *direct* dependencies only.** BayesBase itself pulls in
`Distributions`, `DomainSets`, `SpecialFunctions`, `StaticArrays`, `StatsAPI`, `StatsBase`,
`StatsFuns`, `TinyHugeNumbers`, `LinearAlgebra`, `Random`, `Statistics`. So the base's
transitive closure is not small — it just contains no `ExponentialFamily`. Two corrections
to the table above follow: `DomainSets` does **not** move to the approximations package
(BayesBase brings it regardless), and `StatsFuns`/`StatsBase`/`SpecialFunctions`/
`TinyHugeNumbers`/`Distributions` are already in the closure rather than being new
`StandardMessagePassingRules` dependencies.

**Sequencing, to bound the blast radius:**

1. `MessagePassingRulesBase` alone — macros, types, dispatch, registry. No rules. Ends with
   the go/no-go gate below.
2. `MessagePassingRulesTestUtils` alongside it, early enough that step 3 is written
   test-first.
3. `StandardMessagePassingRules` — standard distribution nodes plus arithmetic (`+`, `-`,
   `*`, dot).
4. `MessagePassingRulesApproximations`, then non-standard nodes spinning out into their own
   packages — Delta, Flow, Autoregressive, GP, BIFM, Pólya (which also resolves the
   licensing conflict), …
5. ReactiveMP engine rewritten against the new base; tooling migrated (ReTestItems, Runic).

This work spans multiple sessions and wants external feedback. Carry it on a long-lived
branch with a repo-level `PLAN.md`; step 1's go/no-go gate is the point where outside
review is most valuable, because it is the one decision that cannot be walked back cheaply.

## Files

- `src/rule.jl` (1958 lines) — replaced. Macros, generic functions, error machinery,
  `@test_rules`. Delete `showerror` string decoding (`:1609-1737`, `:1836-1930`).
- `src/nodes/nodes.jl` — split: traits/registry/`@node` go down, `FactorNode`/`activate!`
  stay in the engine. Relax `prepare_interfaces_check_adjacent_duplicates` (`:241-256`)
  and `prepare_interfaces_check_num_inputarguments` (`:264-270`) for group members while
  keeping the "you passed `x` instead of `x[i]`" diagnostic for non-group interfaces.
- `src/nodes/dependencies.jl` — `__collect_latest_updates` (`:19-33`) must collapse
  consecutive same-name interfaces into one `ManyOf` name; generalise
  `combineLatestMessagesInUpdates` to marginals.
- `src/nodes/predefined/{mixture,normal_mixture,gamma_mixture}.jl` — ~70–85% deleted.
  Retain: aligned-group dependency, construction validation (N≥2, equal lengths,
  mean-field), `ManyOf`-aware entropy. Delete as dead: the three `…NodeFactorisation`
  singletons, `Mixture`'s `RequireMarginal` path (3-arg method unreachable from the 4-arg
  caller; ~130 lines incl. `collect_latest_*`), unreachable `@average_energy Mixture`.
- `src/nodes/predefined/delta/` — heaviest engine coupling; its layout system is more
  custom than the mixtures'.
- `src/message.jl`, `src/marginal.jl` — split value types (down) from observables (engine).
- `src/nodes/equality.jl` — `BitVector` caches → `Vector{Bool}` (bit-packed writes are
  read-modify-write on a shared word; neighbouring indices race).

## Open items

1. **Rule syntax final form** — `m[x]`/`q[x]` settled in principle; outbound-edge spelling
   and where `algorithm` sits in the header are not.
2. **`aligned` generality** — everything in-tree is `k ↔ k`. `q[p[f(k)]]` extends
   naturally; don't build until something needs it.
3. **Per-(target, factorisation) group selection** — assumed per-target; all four in-tree
   cases work because the mixtures pin their factorisation. One-way door in the syntax.
4. **Ruleset axis** (`StandardRules()`, `Overlay(mine, standard)`) — separate from
   `algorithm`, would give scoped rule tables and stop downstream packages invalidating
   each other's inferred call sites. Not requested; `algorithm` may already suffice.
5. **Reactant and StableCholesky** — deferred to their own effort. Per-rule compilation is
   an explicitly supported *research* path when it happens, not a rejected one; whole-sweep
   tracing and `vmap` are a superset of it, not a competing approach. Nothing to decide
   here beyond keeping `buffer_like` extensible.
6. **`reverse(...)` in the mixture marginal wiring** — undocumented, means/precs swapped.
   Pin current behaviour with a regression test before touching.
7. **`EdgeLabel.index`** exists in GraphPPL but RxInfer discards it; ReactiveMP re-derives
   group indices from position, silently depending on neighbour order. Plumb it through.
8. ~~**Does `MessagePassingApproximations` exist at all?**~~ **RESOLVED.** Yes, as
   `MessagePassingRulesApproximations`, holding `Unscented`/`Linearization`/`CVI`/`smoothRTS`
   — but as **standalone numerical utilities that do not depend on the base package**, not as
   algorithms. The unused methods are deleted rather than packaged. See § Approximations are
   utilities, not algorithms.

## Migration

- 491 rules + 172 `@call_rule`/`@test_rules` sites in tests. Build the transform on
  **JuliaSyntax** (source-preserving green tree), not regex and not MacroTools — bodies
  contain arbitrary code, `where` clauses and comments worth keeping.
- Run the tool with ReactiveMP v6 loaded so it can call `interfaces(fform)` as an
  **oracle** to decide whether `y_x` is one interface or the cluster `(:y, :x)`. Without it
  you are regex-guessing on `_`, which is the exact bug class being deleted.
- Migrate per rule directory (52 of them), reviewing diffs directory-by-directory.
  `@test_rules` gives per-rule numerical regression coverage for free.
- Hand-written: `mixture/switch.jl`, the ~15 rules touching raw `messages[i]`/`marginals[i]`
  tuples, the 5 `MessageMapping` construction sites and 4 delta layout files.
- Canary: `NormalMixture((:m, k))` — indexed target + `ManyOf` marginals + `where {N}` +
  aligned group dependency in one rule.

## Migration guide (published, for downstream authors)

Distinct from the internal transform above. That is a one-off tool we run over our own 390
rules; **this is a durable, published document** for anyone maintaining their own nodes and
rules — RxGP, and colleagues with custom rules in their own codebases. It must work for a
human reading it *and* for an AI agent pointed at it, since that is how much of the
downstream migration will actually happen.

Lives as `MIGRATION.md` in `MessagePassingRulesBase`, surfaced in its docs and linked from
ReactiveMP and RxInfer.

**Requirements that make it usable by an agent, not just readable:**

- **Mechanical before/after pairs, not prose.** Every v6 construct maps to its v7 form as a
  concrete pair. An agent should not have to infer the rule from a description.
- **Complete case coverage**, including the fiddly ones — `ManyOf` → variadic groups,
  indexed edges `(:in, k)`, joint marginals `q_y_x` → `q[y, x]`, `meta` → `algorithm`,
  `@logscale`, `getnode`/`getnodefn`, `Marginalisation` removal, the renamed macros.
- **An explicit "cannot be translated mechanically" section.** Rules touching raw
  `messages[i]`/`marginals[i]` tuples, rules constructing graph objects, anything relying on
  `meta` as a mutable workspace (the RxGP `GPCache` pattern). An agent must be told to stop
  and ask rather than guess — silently-wrong rules are the worst outcome here.
- **A verification procedure, not just a rewrite procedure.** Ship a checker in
  `MessagePassingRulesTestUtils` that runs the v6 and v7 rule on identical inputs and
  asserts agreement. This is what makes agent-driven migration trustworthy rather than
  hopeful: an agent that can check its own work is a different proposition from one that can
  only pattern-match. Node-definition verification is the stronger form where it applies.
- **Executable examples.** Every before/after pair is a doctest that CI runs. Migration
  guides rot precisely because they are written once against a design that then moves;
  `doctest = true` is already a decision, so this costs nothing and prevents drift.
- **A short preamble addressed to an agent** — what to read first, what to never guess at,
  how to verify, when to stop and ask.

**Write it during the migration, not after.** The mechanical rules get discovered while
porting our own rules; reconstructing them later from memory guarantees the guide is
incomplete in exactly the places that were fiddly. **The tool and the guide should be
derived from one source** — if the transform encodes a rule, the guide documents that same
rule, with a test asserting they agree.

## Testing infrastructure

Applies to **both** the new packages and ReactiveMP itself.

### Tooling

- **Stay on TestItemRunner; fix `runtests.jl` instead.** An earlier draft proposed moving to
  ReTestItems for name/tag filtering. That was unnecessary: **TestItemRunner's filter already
  receives `(filename, name, tags)`** — its own docstring example is
  `filter = ti -> !(:skipci in ti.tags)`. Today's `runtests.jl` merely *chooses* to filter on
  `filename` alone. Name and tag filtering is ~10 lines in `runtests.jl`, not a package swap.
  TestItemRunner is also the julia-vscode-aligned runner, and `@testitem` itself comes from
  TestItems.jl, with VS Code discovering items by scanning source rather than via any runner —
  so the format is portable and this choice stays reversible.
  Still do: tag taxonomy (`:rules`, `:nodes`, `:engine`, `:alloc`, `:slow`, `:quality`),
  `make test` = fast subset / `make test-all` = everything. Precompilation, not execution, is
  the real latency cost in an agent loop.
  **Revisit ReTestItems only if CI wall-clock becomes the bottleneck** — its one real
  advantage is distributed parallel workers, a CI argument rather than an iteration-speed one.
- **Runic** replaces JuliaFormatter. Deterministic and zero-config, which eliminates by
  design the formatter-version drift the current `Makefile` comment documents (CI and
  contributors disagreeing with no code change). Already in use in StableCholesky.jl.
- **Re-enable the two disabled Aqua checks**, plus `deps_compat`'s `check_extras`.
  - `ambiguities`: turn it on, but budget it separately. The key-set argument (rules with
    different input sets provably can't be ambiguous) only applies to the **new** design —
    it cannot justify cleaning up v6's existing ambiguities. **Measure the current count
    first** and treat resolving them as its own task with its own estimate. Registry-based
    checker as primary, Aqua as backstop.
  - `piracies`: **measured — it can be switched on today.** `Aqua.Piracy.hunt(ReactiveMP)`
    reports exactly **3** pirate methods, none of them rules: `default_prod_rule` and
    `prod` for `Uniform`×`Beta` (`nodes/predefined/uniform.jl:6,9`, a deliberate
    mathematical special case that arguably belongs in ExponentialFamily) and a `dot`
    overload for `ForwardDiff.Dual` (`fixes.jl:12`, an explicit upstream hotfix with issue
    links). Move them or list them in `treat_as_own`, and turn the check on.
  - **Caveat, also measured: the piracy check is vacuous for rules, and always will be.**
    Aqua's rule is that a `DataType` is foreign only if the type *and every one of its
    parameters* is foreign, and `is_foreign(::Symbol) = false` unconditionally. Verified:
    `is_foreign(Val{:out}, pkg) == false` while `is_foreign(Val{1}, pkg) == true`. Since
    every rule target carries the edge name as a `Symbol` type parameter — `Val{:out}`
    today, `MessageTarget{F, :out, …}` in the new design — no rule can ever be flagged,
    whoever defines it. So enabling the check is worth doing, but **it must not be claimed
    as evidence that the rule-package split is piracy-clean**; it says nothing either way.
    This also removes piracy as an argument for the ruleset axis (open item 4).
- **Expand JET well beyond its current two uses.** It is the tool that enforces the
  devirtualization gate; a JET assertion that rule invocation is free of dynamic dispatch is
  what catches a regression in the core design months later.

### The `@test_rules` successor

Lives in **its own package, `MessagePassingRulesTestUtils`**, not in `src/` as it does
today. Consumers — `StandardMessagePassingRules`, a hypothetical `AutoregressiveNode` —
list it under `[extras]` and the `test` target, so its dependencies (quadrature, sampling,
possibly Turing) never reach a node package's runtime. No weakdeps and no package
extensions are involved anywhere in this.

This also drops the "`Test` must be imported globally and `@test` called by name" coupling
and the 4-argument callback form that exists only to work around it.

Keep the table-driven shape. One table checks:

1. numerical output (today's behaviour)
2. **type promotion — now on by default.** Not a float-width check: it is a container- and
   number-type propagation check, and it is what makes `ForwardDiff.Dual` work through a
   rule. Slow variants tagged `:slow`.
3. `rule` and `rule!` agree, when `inplace = true`
4. *optionally*, that the rule is non-allocating — a separate opt-in flag, **not** implied
   by `inplace`

Not checked: purity (see above).

### Verification against the node definition

Worth building, and the strongest available answer to a real weakness: the existing tables
are **golden-value tests** that lock in whatever a rule produced the day it was written,
bugs included — a rule wrong from birth has a passing test forever.

The machinery already exists. `@node` generates `nodefunction` (the node's logpdf);
`src/approximations/` has `ghcubature`, `srcubature` and importance sampling;
`rules/fallbacks.jl` already builds an unnormalised logpdf from the node function. So the
reference update can be computed numerically from the node definition — BP as
`∫ f(x) ∏_{j≠i} m_j dx_{≠i}`, naive VMP as `exp(E_{q(¬i)}[log f])` — and compared to the
analytic rule. This tests the *maths*, not a regression table, and is what makes porting
390 rules credible.

A sampling-based variant (e.g. building the local factor as a Turing model) is an **idea,
opt-in behind its own flag**, not a default. It has one subtlety that must be deliberate or
the test silently measures the wrong thing: a BP message is not a posterior. Sampling the
local factor yields a *marginal*, whereas the message is that integral with no prior on the
target edge. So the comparison must be: rule message × a known proper test prior on the
target edge, against the MCMC marginal obtained under that same prior. Done naively it
appears to work on symmetric conjugate cases and fails confusingly elsewhere.

Scope honestly: stochastic nodes only (`nodefunction` is not generated for deterministic
ones); MC error forces loose tolerances and quadrature hits dimensionality fast; rules that
are themselves approximations (delta, CVI, projection) have no exact reference; improper or
unnormalised messages, `PointMass` inputs, and rules returning `ProductOf`/`FactorizedJoint`
need special handling. A property test complementing the tables, tagged `:slow`.

### Strict TDD

- **Failing test first in every PR**, unless explicitly justified in the PR why it is not
  possible or not required.
- **Registry-backed coverage, mechanically enforced:** because `@rule`/`@node` emit data,
  "every rule has a test" becomes a CI check — cross-reference the registry against tested
  rules and fail on any `RuleSpec`/`NodeSpec` with no test entry. Nothing in the current
  system can do this, since rules exist only as methods.
- Coverage floor, fail on decrease.

## Documentation

Docstrings + jldoctests + Documenter `@example` blocks. Enough, not more.

- CI runs the `@example` blocks and doctests the docstrings (`doctest = true`).
- **Everything carrying a docstring must appear in the documentation**, including
  unexported items, qualified (`ReactiveMP.documented_function`). Enforced by Documenter's
  `checkdocs = :all`, which errors on any docstring not included — so the rule is a CI
  failure, not a convention.
- Genuinely internal helpers use **comments, not docstrings**.

### Layering across packages

The educational and interactive story (see that section) needs documenting at three
levels, each aimed at a different reader. Largely **later work**, but recorded now because
it shapes where things are written rather than being bolted on afterwards:

- **`MessagePassingRulesBase`** — low-level reference. What a rule *is*, the anatomy of a
  definition, how to invoke one by hand, how to query the registry, how to write your own
  node and rules. The "I want to add a node" audience.
- **ReactiveMP** — engine level. How rules are scheduled and wired, dependencies and
  factorisation, free energy, where messages actually flow. The "I want to understand the
  machinery" audience.
- **RxInfer** — high-level, the umbrella package and the widest audience. Conceptual and
  tutorial framing: what message passing is, worked examples, the classroom material. Links
  down for detail rather than duplicating it.

Avoid restating the same content at each level — the failure mode is three
half-maintained copies that drift.

## Verification

- `@test_rules` numerical regression per rule, unchanged semantics, run per directory
  during migration.
- **Go/no-go gate before writing any rules:** assert with `@code_typed`/JET that the new
  dispatch path compiles to the same code as a direct call, and that a rule invocation is
  free of dynamic dispatch. If the indirection doesn't devirtualize, the design is wrong.
- `check_rules()` + `check_rule_ambiguities()` + registry/method-table consistency in CI.
- Allocation regressions, where a rule opts into the non-allocating flag: kernel (`== 0`),
  rule with a provided buffer (`== 0`), full sweep (golden number + tolerance) — three
  distinct levels, never conflated, and none of them implied by `inplace`. Existing
  precedents: `test/annotations_tests.jl:107`,
  `test/rules/mv_normal_mean_scale_precision/out_tests.jl:129-136`.
- Buffer-escape detection: run the full suite in checked mode as a separate CI job.
- Mixture rewrite: pin current behaviour (including the `reverse` quirk) with regression
  tests first, then delete.
- End-to-end: RxInfer's test suite and RxInferExamples against the new packages.
