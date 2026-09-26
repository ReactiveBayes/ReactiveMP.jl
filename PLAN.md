# Ground-up rewrite of the ReactiveMP rule/node system

## Context

ReactiveMP.jl is currently "a can of everything": one package holding the message-passing
engine, 45 node definitions, 384 `@rule`s, 106 `@marginalrule`s and 57 `@average_energy`s
(~24k SLOC). The rule/node layer is the oldest part of the codebase and it shows.

These counts cover `src/` only; the 4 `CVIProjection` rules in `ext/ReactiveMPProjectionExt`
(2 `@rule`, 2 `@marginalrule`) are counted with that extension, not in the 490.

(Earlier drafts said 48/390/108. Those figures counted the docstring examples inside
`src/rule.jl` and `src/nodes/nodes.jl` alongside the real definitions. Harmless as prose,
but Phase 4's registry-backed coverage check counts definitions, so the corrected numbers
are used here. Likewise there are **50** rule directories, not 52.)

Goals: extract the rule/node layer into standalone packages so `AutoregressiveNode`,
`RxGP` etc. can exist independently of the engine; and, at the same time, redesign that
layer from the ground up rather than porting it. Clean break, new major version of
everything, no backwards compatibility. **The Julia floor stays at 1.10** — nothing in
this design requires a newer one (see Dispatch axes on why `ScopedValues` is not used),
so the floor moves only when something concrete needs it to. *(Superseded in Phase 4.5, user: work targets 1.13
only, wired with `[sources]`; the floor is revisited at registration. See § Repository layout.)*

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
  throwaway `randomvar` inside a rule to reach product-with-log-scale machinery. *(The
  switch rule computes the product's log scale itself, under `MixtureBP(; prod = GenericProd())`,
  `DISCUSSION.md` §3.50.)*

## Architecture

### Dispatch axes

Rules dispatch on: **node**, **target**, **algorithm**, **inputs**, plus a non-dispatching
**context**.

- **`algorithm` replaces `meta`.** `meta` today is already two things fused: an algorithm
  selector (`Unscented` vs `Linearization` vs `CVI`) and a parameter bag (AR order,
  kernels). Naming it `algorithm`, giving it a default and making it swappable absorbs
  `Marginalisation`/`MomentMatching` — that axis disappears rather than being deleted
  separately. It also absorbs `RequireMessage`/`RequireMarginal`/
  `RequireEverythingFunctionalDependencies` (see Dependencies), which are **deleted**, not
  ported (user, Phase 4.5). They bundled three things, each of which now has its own home:
  - *what a node's rules consume* is declared on the node's algorithm. The two in-tree users
    already carry one: Probit's `ProbitMeta(p)` (its moment matching) and ContinuousTransition's
    `CTMeta(transformation)` become those nodes' own algorithms, with their dependencies
    declared on them;
  - *a per-model override for one node*, v6's `where { dependencies = … }`, becomes choosing an
    algorithm for that node: a `DefaultAlgorithmExtension` that declares different
    dependencies and inherits every rule;
  - *an initial value*, such as Probit's `in = NormalMeanPrecision(0, 100)` for a rule that
    depends on its own edge, is **initialization**, not a dependency. A node may declare a
    default initial message on its definition, separately from `dependencies`: Probit's
    `@define_factor_node` has `initial_messages = [:in => NormalMeanPrecision(0.0, 100.0)]`,
    used where the model sets none.
  Mixture's `RequireMarginal` path was unreachable in v6 and goes with them. The user
  documentation is written in Phase 5: a page on declaring dependencies in the new terms
  only, and a `MIGRATION.md` section mapping the old types to the new pieces. *(The guide is
  the docs page `docs/src/migration-guides/v6-to-v7.md`, not a `MIGRATION.md`, `DISCUSSION.md` §3.36; both were
  written in Phase 5 step 9.)*
- **There is one algorithm, `DefaultAlgorithm()`, and it is not an inference scheme.**
  Bethe free energy minimisation. Belief propagation, variational message passing and their
  structured forms all come from the **factorisation**, through the engine's default
  dependency scheme: a rule gets the messages inside its own cluster and the marginals of the
  other clusters. Every node runs under `DefaultAlgorithm` unless it declares otherwise, and
  its rules name no algorithm. A custom algorithm exists for exactly two reasons: a **rule
  switcher**, when someone wants a different set of rules, or a **node's own algorithm**, for
  a node that genuinely needs one, such as Delta (its approximation method and inverse),
  Autoregressive (its order) or the mixtures (always variational, whatever the factorisation).
  *(Until the Phase 4.5 reconciliation this was misread as `BP` and `VMP` algorithms; see
  `DISCUSSION.md` §3.20.)*
- **Two kinds of custom algorithm.** A direct subtype of `AbstractAlgorithm` **stands alone**:
  only its own rules and dependencies apply. A subtype of `DefaultAlgorithmExtension`
  **extends the default**: resolution looks for its own rule first and falls back to the
  default's, and likewise for dependencies. The fallback is a second lookup in the untyped
  `find_*` fallbacks, not dispatch on a supertype, which would make an override with broader
  inputs ambiguous with the default rule it replaces. An inherited rule runs with
  `DefaultAlgorithm()` in its `algo` slot, the algorithm it was written for
  (`rule_algorithm(spec, algorithm)` gives the engine that value).
- **`context` (`ctx`) is infrastructure, never dispatched on** (`DISCUSSION.md` §3.49): a
  `RuleContext`, a mutable holder of a typed `NamedTuple` of services, passed by reference and
  read-only to rules; every read `ctx.name` is inferred. The engine supplies `node` (the node
  itself), `rng` (the task's generator) and `matrix_correction` (`nothing`, each rule applying
  its own default) (`src/context.jl`), merged with the activation option `context`, any
  `NamedTuple`, which adds or overrides services. Any name is allowed, so a rule may need a
  service of its own; a rule declaring one nobody supplies is an error when the engine resolves
  it. There is no `linalg` service
  (#13) and no `product` service (§3.50). Output buffers and scratch are body slots, not
  services.

  **`ctx` does not carry annotations.** Annotations are
  their own body slot, `ann`, which the rule both reads (the annotations that *arrived* with
  its inputs) and writes (`annotate!`); context is state the rule only reads. They
  are separate slots, and merging them is a category error. The engine builds one context per
  node at activation and shares it among the node's mappings — not a global and not a
  `ScopedValue`, which was never necessary, since a plain default argument does the same job.
- Rules declare which context **services** they need (e.g. `ctx = (:rng, :matrix_correction)`);
  `missing_services(spec, ctx)` lists the declared ones a context does not supply, and the
  engine calls `check_services(spec, ctx)` as it resolves each rule, message, marginal and
  average energy, an error naming the rule and the services. The interactive calls do not
  check: they run with the context their caller passes, empty by default.

### Rule surface

Role lives in a container, not a name prefix. `m[...]` and `q[...]` are two containers;
**declaration and body use the same key**, so no name is ever derived and nothing is ever
split. Interface names may contain `_` again.

Everything is a **keyword**, and the body is an **ordinary Julia lambda over a real
arguments object** — not a body the macro rewrites. One form, used by every rule.

```julia
@define_message_update_rule(
    node    = NormalMeanVariance,
    target = :out,
    args    = (m[:μ]::PointMass, m[:v]::PointMass),
    body    = (args) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v])),
)
```

With a node's own algorithm, a variadic group and a log scale (`DISCUSSION.md` §3.50):

```julia
@define_message_update_rule(
    node           = Mixture,
    target         = :switch,
    algorithm      = MixtureBP,
    args           = (m[:out]::Any, m[:inputs...]::Any),
    reads_logscale = true,          # reads args.logscale.m[...]
    logscale       = from_body,     # or a constant, `logscale = 0`, or `(args) -> ...`
    body           = (algo, args) -> begin
        ...
        with_logscale(Categorical(softmax(ls)), logsumexp(ls))
    end,
)
```

The log scale is part of the message, not an annotation: the scalar with
`message = exp(logscale) · result` for the normalised result a rule returns. A rule declares it
statically; one that omits it declares none, and its message's log scale is an
`UndefinedLogScale` that propagates; only `require_logscale` errors on it, where a number is
needed. The engine tracks log scales only under the activation option `logscales = true`;
otherwise messages carry `nothing`. Every public call of a rule returns a `RuleResult`
(`getresult`, `getlogscale`, `getrule`, …); the engine never builds one.

**Why symbols.** In a real lambda, `args.m[μ]` is an `UndefVarError` — `μ` is not a
variable. Only `args.m[:μ]` works, so the body is *forced* to symbols; the declaration
follows, because declaration/body agreement is the whole reason `m[]`/`q[]` was adopted over
name mangling; and consistency carries symbols into `target` and into `@define_factor_node`.
The colon is therefore load-bearing, not decoration. Measured: `args.m[:μ]` on a
NamedTuple-backed container is type-stable and allocation-free.

- `m[:μ]` and `q[:μ]` in one signature is unremarkable — different containers.
- `q[:y, :x]` is a structural cluster; `q[:p][k]` is a member of the group `p`. **Not**
  `q[:p[:k]]`, which parses as `(:p)[:k]` — indexing a `Symbol`.
- **A cluster is the tuple of its members**: `q[(:y, :x)]`, with `q[:y, :x]` as shorthand.
  Inside a cluster a group's name means all of its members jointly, so `q[(:in,)]` is
  Delta's joint over its inputs (v6's `q_ins`) and `q[:out, :in]` mixes a single interface
  with a group. A marginal rule computing it is `target = (:in,)`. `q[:in]` stays the tuple
  of the members' own marginals. A one-member cluster of a *single* interface is rejected
  rather than accepted as a second spelling of `q[:μ]`. Added in Phase 4.5; the access runs
  through the same type-level `Val` key, measured allocation-free in `gate:containers`.
- **No symbol is ever built at run time.** A joint is keyed by the *tuple of member symbols
  carried in the type*: `q[:y, :x]` is an `@inline` forwarder to `getindex(q, Val((:y, :x)))`,
  constant propagation of the literal symbols makes that `Val` static, and a `@generated`
  lookup on the container's key-tuple parameter resolves it to a `getfield` by position.
  Forming `Symbol("y,x")` or similar is ruled out — it is slow and it is mangling again.
  Members are listed in interface-declaration order (#9). Single keys sit in a NamedTuple
  in **canonical sorted order**, applied by the macro to the dispatch signature and by a
  `@generated` constructor to the arguments, so the key order is fixed at compile time and
  needs no node lookup at macro expansion. Measured at Phase 3 step 2 on 1.10.12 and 1.13:
  `@inferred` and allocation-free, so the explicit `args.q[Val((:y, :x))]` fallback spelling
  was not needed (`DISCUSSION.md` §3.16).
- `m[:inputs...]` is a variadic group, replacing `ManyOf{N,T}` plus its `where {N}`.
  Verified to parse.
- The outbound target is `target = :out`, or `target = (:m, k)` for a member of a group.
  *(The keyword was `towards` until 2026-09-23, when the user renamed it: a marginal rule's
  target is a cluster, and nothing is sent "towards" it. The rename covers every rule
  macro, the interactive `call_*`/`which_*` functions and the test tooling.)*
- Message, marginal and average-energy definitions retain their distinct roles and return
  contracts. They lower onto **three separate generic functions** (see Naming) that share
  one registry, one error path, one ambiguity checker and one test macro — shared
  *infrastructure*, not a single dispatch function. `@define_average_energy` gains the
  `algorithm` axis.

**Body slots.** The body declares only the parameters it needs, by name, in this canonical
order:

```
(output, scratch, algo, ctx, args, ann)
```

(An earlier form had a slot `node`. It was dropped at the Phase 3 sign-off: nothing
dispatches on the node, so it lives in the context as `ctx.node`, and a delta rule reaches
its function as `getnodefn(ctx.node, …)`. See open item #12.)

`scratch` is a rule's working memory (`DISCUSSION.md` §3.44): the rule declares how to build it
from its inputs, `scratch = (algo, ctx, args) -> …`, and takes it as a slot; the engine keeps
one per outbound stream and reuses it. It is write-before-read, carries nothing between calls,
and is never shared between a node's rules, so a rule using it stays pure. It is separate from
`inplace`, and the two combine.

The macro reads the names written in the lambda and fills the rest. Order is enforced, so
every rule reads the same way down a file; relaxing that later is easy, tightening it would
be breaking. An unrecognised name is an error naming the valid set. `output` appears exactly
when `inplace = true`.

**The target is threaded to every body, but is not one of the slots.** An indexed target
`target = (:m, k)` binds `k` in the body, and `k` is a runtime value the lowered body cannot
close over — so the call carries the target and the macro emits `k = index(target)` as an
ordinary binding when the declaration names an index. Writing `k` is how you ask for it;
there is no `target` slot to request. This mirrors v6, which injects `k = on[2]` at macro
expansion. Found in Phase 0.

`algo` is **read access to the algorithm value** (AR order, kernels, inducing points — the
parameter-bag half of the old `meta`). It is *not* the dispatch mechanism: dispatch is the
`algorithm` keyword, which types the generated method's algorithm argument. A rule that
omits `algorithm` belongs to its node's default, `DefaultAlgorithm` for almost every node.

`ctx` and `ann` are deliberately separate: `ctx` is infrastructure the rule only
*reads*; `ann` carries annotations in **both** directions — the incoming ones, keyed exactly
like the inputs (`ann.m[:out]`, `ann.q[:μ]`), and the outgoing sink the rule writes with
`annotate!(ann, key, v)`. Annotations never take part in dispatch: they must not select
the mathematics. Merging `ann` into `ctx` is a category error. A log scale is not an annotation
(`DISCUSSION.md` §3.50): it is declared with `logscale` and read as `args.logscale.m[:out]`.

```julia
body = (args, ann) -> begin
    count = getannotation(ann.m[:out], :input_count)   # incoming, arrived with m[:out]
    ...
    annotate!(ann, :input_count, count + 1)            # outgoing
    result
end
```

`@define_factor_node` uses the same vocabulary:

```julia
@define_factor_node(
    node       = Mixture,
    type       = Stochastic,
    interfaces = [:out, :switch, :inputs...],
    algorithm  = MixtureBP,    # omitted for almost every node: DefaultAlgorithm
)
```

Almost every node omits `algorithm` and runs under `DefaultAlgorithm`, so the ~350 rules
omit it too; only a rule switcher's or a node's own algorithm is named.

**The cost, stated plainly.** A one-line rule becomes roughly six, across ~490 definitions.
That is accepted deliberately in exchange for one uniform surface with no in-body macros.
And the `args = (...)` declaration is still a mini-language the macro parses — it is a
signature, so that is unavoidable. "No magic" holds **in the body**, not everywhere.

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

**In-body macros — gone entirely.** An earlier design had `@allocate` and `@logscale` as
short, unexported macros that only worked inside a rule body. They were not really macros:
they were tokens the enclosing definition macro rewrote, which is why using one elsewhere
failed confusingly. The keyword form removes the category rather than renaming it —
`@allocate` is the `preallocate` keyword, and `@logscale` is the `logscale` keyword: a
constant, a function of the body's slots, or `from_body` with the body returning
`with_logscale(result, logscale)` (`DISCUSSION.md` §3.50).

**Invocation macros — exported, and named after what they invoke** (user, Phase 3):
`@call_message_update_rule`, `@call_marginal_update_rule`, `@call_average_energy`, with
`@which_*` counterparts and function forms of the same names. An earlier draft kept them
short (`@call_rule`) because they are typed often at a REPL; the user reversed that, so that
each invocation mirrors its definition macro and nothing is left generic.

### Educational and interactive use

A first-class goal, not a by-product, and **built in full in Phase 3** — invocation macros,
registry queries, `@which_message_update_rule` and its siblings, the coverage matrix and `text/plain`/`text/html` display
(decided while planning Phase 3). The system is taught with in the BMLIP course at
TU/e, where invoking rules by hand is a good way to show what message passing actually
does. The registry is what makes all of this cheap — today's `print_rules_table()` scrapes
`methods()` through `arg_decl_parts` string offsets, which is why nothing better was ever
built on it.

- **Manual rule invocation stays first-class.** `@call_message_update_rule`,
  `@call_marginal_update_rule` and `@call_average_energy` (v6 could not call an average
  energy by hand at all), each with a function form of the same name.
- **Querying the registry**: `list_rules(NormalMeanVariance)`, `list_rules(NormalMeanVariance, :out)`,
  filtering by algorithm, and `@which_message_update_rule` (which `RuleSpec` fires for these inputs, with
  source location). "Which factorisations does this node support?" becomes a query: the
  rules' declared inputs say which combinations of messages, marginals and clusters exist.
- **Rule coverage matrix** — edges × rule sets (the default and any custom algorithm) for a
  node, cells showing which rules exist. Simultaneously a teaching artifact and a
  development one ("what is missing in the port?").
- **Rich display via `Base.show` MIME methods** — `text/plain` for the REPL, `text/html` for
  notebooks, since the course uses them. Zero dependencies, works everywhere.
- **Visualisation via the extension mechanism**, following GraphPPL's pattern: diagrams of a
  node's interfaces/groups, of what a rule receives under a given factorisation, and of the
  coverage matrix activate when a plotting package is loaded, and otherwise fail with a
  descriptive "load X to enable this" message rather than a `MethodError`.
- **Rich display of `RuleResult`** (user, `DISCUSSION.md` §3.50, §3.51; built for a single rule call).
  What a call returns can show itself, through Julia's multimedia `show(io, mime, x)`: a
  `text/plain` report in the terminal (the node and the target with its direction, the inputs
  with their types and values, marked `m` or `q`, the incoming log scales, the algorithm and its
  parameters, the context services the rule declared, the scratch, the result, the log scale and
  where it came from, the rule's signature and source), in colour only when the stream has
  `:color`, respecting `:compact` and `:limit`; and `text/html` for Jupyter, Pluto and
  Documenter's `@example` blocks: a self-contained card, inline CSS following the reader's theme,
  with an inline SVG of the node — its interfaces as edges, the inputs used as incoming arrows
  styled as messages or marginals, the unused interfaces greyed, the target the highlighted
  outgoing arrow — and the report's sections as tables. Dependency-free; `visualize_spec` stays
  the extension point for richer backends. Not in it (user, §3.51): traces of an inference run,
  and the maths of a node or a rule, which nothing declares.
- **Error messages are pedagogy.** The "no rule found" output showing near-miss rules with
  per-slot ✓/✗ diffs is a teaching tool for students who get an input wrong, which is the
  common case in a classroom.

### Registry, errors, introspection

`@define_message_update_rule`/`@define_factor_node` emit **data** (a `RuleSpec`/`NodeSpec`)
alongside the method, stored as a per-module `const` and discovered by scanning loaded modules. Do **not** `push!` into a
ReactiveMP-owned global — a downstream package's top-level `push!` runs during *its*
precompile and lands only in its own image.

**Two mechanisms, not one** (clarified with the user after Phase 4.5 step 4, `DISCUSSION.md`
§3.23, kept as is). *Lookup* is the base package's method table: each definition adds a method
to `find_message_rule`/`find_marginal_rule`/`find_average_energy`, so every loaded rule, from
any package, is in one global table resolved by static dispatch. The per-module *registry* is
**introspection only** (listings, `check_rules`, ambiguity checks, coverage, near-miss errors),
and the engine never reads it. No registry keyword and no registry dispatch axis.

This replaces `arg_decl_parts` string decoding entirely: "no rule found" has the target,
inputs and algorithm as live values. Distinguish the two failure classes that are
currently indistinguishable — *no rule of this shape* (wrong dependency/factorisation)
vs *type mismatch* (rule exists, arguments don't fit). Add `check_rules()` (validates
specs against node specs, and dependency specs against rule signatures) and
`check_rule_ambiguities()` (group by input-name set, then Julia's `Base.isambiguous` on each
pair's own methods — a hand-made `typeintersect` reports every disjoint pair, since the
intersection of two disjoint input tuples is a valid, empty type rather than `Union{}`).

Removing the macro-expansion-time registry query means `@define_factor_node` and the rule
definition macros may appear in any order, in any package, including weak-dep extensions —
with one exception: a rule that omits `algorithm` reads its node's default at load time, so it
must load after the node.

**The `RuleSpec` is the execution vehicle, not only registry data.** Dispatch at the call
site resolves `(node, target, algorithm, args)` to a `RuleSpec`, and the spec knows how to
invoke itself: it **stores the body and the `preallocate` lambda**, knows whether it is
in-place and therefore whether to preallocate, which body slots the body requested, whether
it is pure, and which context services it needs. The engine does not branch on `inplace` —
it hands the spec its arguments and the spec resolves the path. `rule(...)` is "resolve,
preallocate, run"; `rule!(buffer, ...)` is "resolve, run".

One object then serves both purposes: what the registry stores for `@which_message_update_rule`, the
coverage matrix and `check_rules()` is the same thing that runs, so the two cannot drift
apart. It can also carry **the body's source text, file and line**, which is what makes
`@which_message_update_rule` able to *show you the rule* rather than merely name it — directly serving the
educational and introspection goals above.

**`RuleSpec` carries no type parameters.** It is a plain immutable struct, every field
ordinary. A sketch; the definition, `lib/MessagePassingRulesBase/src/rulespec.jl`, also has
`kind`, `inputs`, `scratch`, `default`, `services`, `logscale` and `reads_logscale`:

```julia
struct RuleSpec
    body::Function         # the lambda from `body = ...`
    prealloc::Function     # the lambda from `preallocate = ...`, or `nothing`
    inplace::Bool
    pure::Bool
    source::String         # body text, for `@which_message_update_rule`
    file::Symbol
    line::Int
    # + node, target, algorithm, signature, required services, …
end
```

**The decision is deliberate, and its point is `find_rule`.** A spec parameterised on its
body and allocator types would make every rule a distinct `RuleSpec{B, P}`, so a lookup that
cannot statically pin down which rule fires returns a *union* of spec types rather than one
type. Julia union-splits small unions and gives up past a handful of arms, and the failure is
silent. With no parameters there is exactly one `RuleSpec` type, `find_rule` is type-stable by
construction, and nothing downstream has to predict a `Bool` or a closure type in order to
name the type it is holding. Flags in signatures are a well-known source of downstream
instability, and the same objection applies to hoisting the body's type.

The cost is one indirect call per invocation where the compiler cannot see which body is in
the field. **Accepted for now, to be revisited with real rules rather than a toy**: the
alternatives — parameterising on the body, or emitting a separate generated method for
execution and keeping the spec as pure data — are both recorded in `DISCUSSION.md` §3.14 with
their measurements, and either can be adopted later without changing the macro surface, which
is what actually matters. Phase 0 measures the parameter-free representation under the real
spec and reports what it costs; that is a number to act on, not a reason to pre-optimise the
design now.

**What stays true regardless of the representation:** resolution must not go through a
runtime container. A spec fetched from a `Dict` keyed on runtime values infers as `Any`
whatever the spec's own type, so resolution by dispatch on the base package's `find_*`
methods is a requirement, not a preference. The per-module `const` of § Registry is
introspection only and is never read on the resolution path (`DISCUSSION.md` §3.23).

**Measuring this is easy to get wrong, in the direction that flatters whatever you built.**
If the spec is constructed inline inside an inlinable `find_rule`, the compiler constant-folds
the whole expression and *every* representation reports zero allocations — so an implementer
can "confirm" a gate and learn nothing. Equally, a micro-benchmark where a call site can only
ever reach one rule measures the best case and hides the indirect call entirely. So **Phase
0's devirtualization gate must test through the spec**, not merely through dispatch, and must
report the figure for a call site that can reach more than one rule as well as one that
cannot. Record both numbers rather than a verdict. `DISCUSSION.md` §3.14 carries the runnable
comparison across representations.

### Dependencies as a language

Dependencies become a **property of the algorithm**, not the node. Under `DefaultAlgorithm`
they follow the factorisation and are rarely declared; a node whose rules need something
else declares its own algorithm and its dependencies. Written in the same `m[]`/`q[]`
vocabulary as rules:

```julia
@define_factor_node(
    node       = NormalMixture,
    type       = Stochastic,
    interfaces = [:out, :switch, :m..., :p...],
    algorithm  = NormalMixtureVMP,
    dependencies = [
        (:m, k) => (q[:out], q[:switch], q[:p][k]),
        (:p, k) => (q[:out], q[:switch], q[:m][k]),
    ],
)
```

The left-hand side is a **target** — `:out`, or `(:m, k)` for a member of a group — spelled
exactly as `target` spells it. The right-hand side is a tuple of container lookups, spelled
exactly as a rule's `args` spells them.

**The order of the right-hand side is the schedule.** The engine subscribes to a target's
inputs in the order they are declared, and in variational message passing that order decides
which update a rule sees first. It changes how fast a node converges, not where to
(`DISCUSSION.md` §3.24); `NormalMixture` lists its precisions before its means for that reason.

A declaration may write `default` among a target's inputs, `:a => (default, q[:a])`: the
default scheme's inputs plus the listed ones, in interface order (§3.41). A rule may likewise
take `default` among its arguments, whatever inputs the factorisation delivers, beside typed
ones; a joint may hold part of a group, keyed with its members, `(:out, (:T, 1))` (§3.45).
A node declares what it requires of the graph, `matched_groups`, `min_group_length` and
`factorisation = :meanfield`, and `factornode` checks them (§3.38).

For the default dependency scheme, two axes separate:

- **Role** (message vs marginal) stays derived from the factorisation, as today.
- **Group selection** is the new declarative axis. Custom algorithms also need the belief
  and scoring contract described below.

A group selector maps the target index to a tuple of source indices. The four observed
modes are four values of one type; a user lambda is a fifth:

| selector | written | indices | arity | used by |
|---|---|---|---|---|
| `AllGroupMembers` | `m[:in...]` | `1:N` | `N` | `Mixture(:out)` |
| `AllGroupMembersButSelf` | `m[:in][!k]` | `1:N \ {k}` | `N-1` | `DeltaFn((:in,k))` (delta's `TupleTools.deleteat`) |
| `AlignedGroupMember` | `q[:p][k]` | `{k}` | `1` | `NormalMixture((:m,k))` needs `p[k]` only |
| none | not listed | `{}` | `0` | `Mixture((:inputs,k))` |
| `CustomGroupSelector` | `m[:in][select_group_members(f; arity = n)]` | `f(k)` | `n` | — |

**What a rule receives for a group** (decided at Phase 3 step 8): a tuple in member order,
full length, with `nothing` in every position the selection leaves out. So `args.q[:p][k]`
is member `k` whatever the selector, positions keep their meaning, and each placement of
`nothing` is its own concrete tuple type. A rule declares its inputs in the spelling of its
dependencies — `q[:p][k]::T`, `m[:in][!k]::T`, `m[:in...]::T` — and `check_rules()` checks
the two agree.

(Syntax and names decided at Phase 3 step 7. Declared with `@define_dependencies(node, algorithm,
dependencies, free_energy_partition)`, or with `dependencies = [...]` on the node for its default
algorithm. Two engine contracts go with it: **a selection of no members takes no stream and is
already satisfied, and reaches the rule as a tuple of `nothing`s** (`(nothing,)` for one member,
case (d)) — the one-input delta case needs no special branch — and **a singleton
cluster's marginal is the variable's marginal**, which is what delta's `q_out` aliasing was.
Static gating is the node-level `static_inputs = :fold` policy.)

**Hard constraint: selectors must have statically known output arity.** Otherwise the
group tuple's length is runtime-dependent, the rule's signature can't specialise, and dispatch
destabilises downstream. A boundary-dependent lambda like `k -> (k-1,)` violates this at
`k=1`; supported resolutions are a total selector (wrap/clamp/pad) or a distinct target
type for the boundary. Length-unions are unsupported.

The default (same cluster → messages minus self; other clusters → marginals) already
yields `allbutself` and `all` for free. Additional selection and execution semantics must
be tested against the mixture and delta layouts in Phase 0; selector coverage alone does
not establish that the layouts can be replaced.

For custom algorithms, dependencies may constrain the factorisation rather than derive
from it. **The contract (#9, resolved at the Phase 3 sign-off):** requesting `q[:a, :b]` identifies a
joint belief, but an auxiliary belief need not be a separately counted entropy cluster.
Define the consumed beliefs and the entropy partition separately, including ordering and
conflicts with user-supplied factorisation, before freezing the macro surface. Algorithms
outside the standard free-energy construction need an explicit scoring contract.

### Purity

`pure` is declared on the **algorithm**, inherited by its rules, overridable per rule.
`DefaultAlgorithm` is pure, so the ~350 already-pure rules need no annotation; an impure
algorithm carries the marker once rather than once per rule, and a single impure rule declares
`pure = false`, as `CVIProjection`'s joint rule does
(`lib/DeltaMessagePassingRules/ext/DeltaMessagePassingRulesProjectionExt.jl`). BIFM, v6's
offender, is stateless and its rules pure (`DISCUSSION.md` §3.44). Narrowing (pure rule under impure
algorithm) is always safe; **widening must taint the graph-level check**, or the
inheritance is unsound in the direction that matters. Multithreading is not implemented —
the flag exists so it can be, and so impure rules are diagnosable.

**Purity cannot be *proved* by testing** — running a rule twice and comparing outputs tests
determinism, and absence of side effects is not establishable by test. Writing a pure rule
is the author's declared responsibility.

**But the declarations can be audited at run time, and that is what matters in practice.**
See Engine diagnostics: under `check_everything_pure` the engine errors, as each rule is
resolved, naming any rule whose algorithm (or own override) declares impurity
(`DISCUSSION.md` §3.46). Motivating case: when differentiating
through inference, this identifies declared side effects worth investigating. It does not
establish whether a gradient is correct or whether a rule supports differentiation.

A *mutation detector* could also run under that flag — `deepcopy` the algorithm, run,
compare with `==` — catching v6's offender (BIFM mutating its meta). It only sees state
reachable from that object, so it would be a debug aid on top of the audit, not a proof. Not
built; tracked in `PHASES.md`'s not-done table, after the release.

**What `pure = true` actually permits must be written down**, because a rule writes its
`output` buffer and its `scratch` and advances the RNG it reads from `ctx` — so "no mutation"
would outlaw the in-place rules the design exists to enable. Contract: *no mutation of
caller inputs or of shared algorithm state; writes to the rule's own output and scratch are
permitted.* Without that sentence the label is ambiguous exactly where it matters.

**Purity does not establish gradient correctness**, and an earlier draft overstated this.
Local mutation can be perfectly differentiable, and a pure rule can still drop derivative
information or use an unsupported operation. Passing the audit is neither a general
prerequisite for differentiation nor proof of correct gradients. Requiring it in a checked
workflow is a project safety policy.
Add a small set of derivative checks against analytic or finite-difference references,
covering both the allocating and in-place paths.

**RNG ownership — decided at the Phase 3 sign-off.** The RNG comes from `ctx.rng` and is
owned by the caller. An algorithm that carries its own RNG — today `CVIProjection` defaults to
`MersenneTwister(42)` and `BinomialPolyaMeta` to `default_rng()`, neither ever reset — is
`pure = false`. `CVIProjection`'s mutable proposal state makes it impure independently.
The engine supplies the task's `Random.default_rng()` as the `rng` service, overridden through
the activation option `context`, `(rng = StableRNG(…),)` in tests (`DISCUSSION.md` §3.32, §3.49).

### In-place rules

**`inplace` means the result is written into a provided container. It does *not* mean
non-allocating** — intermediates may still allocate. Non-allocating is a stricter,
separate property with its own opt-in test flag.

`rule`/`rule!` mirroring `rand`/`rand!`. Because the allocating form can't be run to
discover a buffer (circular — if only the in-place rule exists there is nothing to run),
and because rewriting a body to `copyto!(out.μ, mexpr)` defeats the purpose (`mexpr`
already allocated), the shape must be declared:

```julia
@define_message_update_rule(
    node        = NormalMixture,
    target     = :out,
    inplace     = true,
    args        = (m[:out]::MvNormalMeanPrecision, q[:m...]::Any, q[:p...]::Any),
    preallocate = (args) -> MvNormalMeanPrecision(
        buffer_like(mean(first(args.q[:m]))),
        buffer_like(cov(first(args.q[:p]))),
    ),
    body = (output::MvNormalMeanPrecision, args) -> begin
        mul!(output.Λ, ...)   # writes straight into the buffer, no temporaries
        output
    end,
)
```

- `preallocate` lowers to `allocate_result(...)`; `body` lowers to `rule!(buffer, ...)`;
  the allocating form is `rule(args...) = rule!(allocate_result(args...), args...)`.
- **The output type is pinned by the ordinary typed lambda parameter** `output::T`, which
  Julia itself enforces (a mismatch is a `MethodError`) and infers. That is stronger than
  the macro-side analysis it replaces.
- **An in-place rule may also request the inbound message on the target edge.** An earlier
  design forbade it, because `m[target]` had to mean either the buffer or the inbound
  message and could not mean both. With `output` as a separate body slot the ambiguity does
  not arise: `output` is the buffer and `args.m[:out]` is the inbound message, and a rule
  may use both.
- `buffer_like(x)` — allocation primitives that **dispatch on the source** (`Vector`→
  `Vector`, `SVector`→`MVector`, `ConcreteRArray`→on-device, …) rather than allocating a
  default and converting, which would repeat the same mistake one level up. Lives in and is
  documented by `MessagePassingRulesBase`, and is **the** extension seam for array types and
  devices — see Reactant below.
- In-place requires writable output fields. `Diagonal`, `SMatrix`, `PDMat`-backed
  (`Wishart` caches a Cholesky), `PointMass`, `ProductOf` are not; a trait gates it and the
  engine falls back to allocating. ~30 entries in BayesBase covers the useful set.
- Default is allocate; reuse is opt-in; a checked mode poisons recycled buffers to expose
  some stale reads. It cannot detect every escape; ownership and retained-value tests are
  required (open item #10). Equality-chain caches hold messages across iterations, so edges
  feeding one need a buffer per cache slot or exclusion from reuse.

Separately and with no ownership reasoning required: decide whether `Message` is immutable
or a `mutable struct` with `const` fields **by benchmark** (Phase 4.5, user: the mutable form
is deliberate, since pass-by-reference can avoid copies), and fold products over raw
distributions rather than `Message`s. Annotations stay the mutable `AnnotationDict` side
channel; typed annotations are not done. The type parameter a message gained is its log
scale, `Message{D, L}`, `L` being `Nothing` when the engine does not track them
(`DISCUSSION.md` §3.48, §3.50).

### Engine diagnostics

A family of opt-in audit flags on the inference engine, **all `false` by default**, all
reporting the *specific offending rule* rather than failing vaguely. Cheap, because the
registry already knows every declaration.

They are fields of `EngineDiagnostics`, the activation option `diagnostics`, checked as each
rule is resolved rather than by a pass over a built graph (`src/diagnostics.jl`;
`DISCUSSION.md` §3.46).

- **`check_everything_pure`** — errors if any rule in the graph declares impurity. For
  differentiating through inference (ForwardDiff), and as the prerequisite audit before
  multithreading is ever switched on. The mutation detector (see Purity) is not built.
- **`check_everything_inplace`** — reports rules with no in-place implementation. A coverage
  audit for latency-sensitive use (robotics, real-time), not a correctness one.
- **`checked_buffers`** — poisons recycled memory with `NaN` before each reuse, as a debugging
  aid. The scratch is the only memory recycled: in-place outputs are allocated per call. A
  stale reference can still observe a valid later value, so pair this with retained-value
  tests and the ownership contract. No CI job runs in checked mode: the engine has no global
  switch for it.

These are also how a user *finds* what to fix: the answer to "why is my model allocating /
why are my gradients wrong / why can't I thread this" should be a rule name, not a hunt.

### Reactant — out of scope here, but the seam is not

Reactant itself and the `StableCholesky` integration are handled separately, **not in this
plan.** What this plan must not foreclose is the seam, which costs nothing now:

`buffer_like` dispatching on the source array type *is* the device seam — one rule body
runs on `Vector`, on `SVector`, and on device arrays with no rule-side change. And in-place
discipline and traceability discipline are nearly the same discipline: both forbid
materialising intermediates, both need static shapes, both need the output container
declared up front rather than discovered from a return value. So `preallocate` +
destination-passing is already the right shape, and `preallocate` is where device placement
would later belong. Keep `buffer_like` extensible and documented; build nothing else.

### Package split

The boundary is already almost clean: rules need only `getdata` and distribution math — no
Rocket, no factor graph, no scheduler. The exceptions were v6's `src/rules/mixture/switch.jl`
(one `randomvar` call; the switch rule computes the product's log scale under
`MixtureBP(; prod)`, `DISCUSSION.md` §3.50) and 12 rules using `getnode`/`getnodefn` (now
`ctx.node`).
**Granularity beyond the packages below — splitting rules by distribution family — is
explicitly deferred.**

Because algorithms are now first-class *values*, the heavy numeric dependencies follow the
algorithms out of the core. Sorting today's 28 deps:

| package | contents | deps |
|---|---|---|
| `MessagePassingRulesBase` | macros, targets, algorithms, argument/annotation containers, context, registry, dependency language, `buffer_like`, `public_equivalent` (§3.29), scratch (§3.44), rule fallbacks (§3.49), log scales, `RuleResult` and its display (§3.50, §3.51) — **not** `Message`/`Marginal`, which stay in the engine | `BayesBase`, `MacroTools` — **and nothing else** |
| `StandardMessagePassingRules` | distribution nodes, arithmetic (`+`, `-`, `*`, dot), logic (`AND`, `OR`, `NOT`, `IMPLY`), the mixtures, `StandaloneDistribution` (§3.47), and the algebra helpers the node packages share (§3.40) | `ExponentialFamily`, `Distributions`, `BayesBase`, `StatsFuns`, `SpecialFunctions`, `LogExpFunctions`, `DomainSets`; `FastCholesky`, `LinearAlgebra` and `MatrixCorrectionTools` from Phase 5 step 5 |
| node packages | one per non-standard node or family (user, Phase 5 step 6; `DISCUSSION.md` §3.30): `GaussianCoupling`, `Probit`, `GCV`, `Autoregressive` (AR, ConjugateAR), `SoftDot` (independent of the AR package, §3.40), `ContinuousTransition`, `Polya`, `BIFM`, `Flow`, `DiscreteTransition`, each `<Name>MessagePassingRules` | each its own |
| `MessagePassingRulesApproximations` | numerical utilities over means and covariances: `Unscented`, `Linearization`, Gauss–Hermite cubature, `approximate_meancov`, `smoothRTS`, shared point/weight machinery (§3.40). **Standalone — does *not* depend on the base package, nor on any distribution package** | `LinearAlgebra`, `FastCholesky`, `FastGaussQuadrature`, `ForwardDiff` |
| `DeltaMessagePassingRules` | the Delta node `DeltaFn{F}`, its algorithm `DeltaApproximation(; method, inverse)`, its dependencies and rules (created in Phase 4.5 case (d)), and `CVIProjection`, its rules an extension on ExponentialFamilyProjection | the base, `MessagePassingRulesApproximations`, `ExponentialFamily`, `Distributions`, `BayesBase` |
| `MessagePassingRulesTestUtils` | all test tooling (see Testing) | quadrature / sampling, whatever verification needs |
| `ReactiveMP` | engine | `Rocket`, `MessagePassingRulesBase`, `BayesBase`, `Distributions`, `LinearAlgebra`, `MacroTools`, `Random`, `TinyHugeNumbers`, `TupleTools`, `UUIDs` |

The base package is genuinely thin. The two single-node-specific deps left the engine with their
nodes: `PolyaGammaHybridSamplers` is a dependency of `PolyaMessagePassingRules` alone, and
`Tullio` of nothing, since the tensor-node DiscreteTransition needs none (§3.45).

**Where the line falls.** `StandardMessagePassingRules` holds what is generic and
model-agnostic — distributions, arithmetic, logic, mixtures. Domain-specific models each get
their own node package (user, Phase 5 step 6, replacing the unnamed `models` package;
`DISCUSSION.md` §3.30): `INVENTORY.md` records them as `node:GCV`, `node:Probit`,
`node:SoftDot` and `node:GaussianCoupling`. A node
leaves for its own package only for a stated reason, and every such reason is recorded in
`INVENTORY.md`: a heavy or licence-bearing dependency (`DiscreteTransition`/Tullio,
Pólya/GPL-3), impurity (`BIFM`, whose v6 rules mutated their meta; its package's rules are
stateless, §3.44), keeping a distribution package
out of the engine and the standard/node split (`Delta`, `DISCUSSION.md` §3.25; the engine
coupling once expected turned out to be the engine owning the node's function), or an
explicit decision (`ContinuousTransition`).

**The full assignment lives in `INVENTORY.md`**, not here: 231 entities — 49 nodes, 165
exported symbols, 8 engine-hook families, 2 extensions and 7 rule-level exceptions — each
with a destination, generated and checked by `scripts/inventory.jl` and gated by the root suite's `:quality`
item. This
section states the policy; the inventory states the 231 decisions, and is the thing to
consult when moving code.

### Repository layout

**Monorepo now, split at Phase 8** (`DISCUSSION.md` §3.40). The new packages live as
subdirectories of this repository under `lib/`, and are promoted to their own
`ReactiveBayes/*` repositories at registration, when compat bounds and CI are set anyway.
Phase 3 froze the base API and Phase 4.5 proved the engine interface; a split before
registration would turn each cross-package change into pull requests and dev pins for
packages nobody can yet install.

```
ReactiveMP.jl/
  Project.toml              # the engine
  src/
  lib/
    MessagePassingRulesBase/
    MessagePassingRulesTestUtils/
    StandardMessagePassingRules/
    MessagePassingRulesApproximations/
    DeltaMessagePassingRules/
    GaussianCouplingMessagePassingRules/
    ProbitMessagePassingRules/
    GCVMessagePassingRules/
    AutoregressiveMessagePassingRules/
    SoftDotMessagePassingRules/
    ContinuousTransitionMessagePassingRules/
    PolyaMessagePassingRules/          # GPL-3
    BIFMMessagePassingRules/
    FlowMessagePassingRules/
    DiscreteTransitionMessagePassingRules/
  compat/v6-comparison/     # ReactiveMP@6.5.0, RxInfer 5.5.2 and the new packages: the v6
                            # oracle, the comparisons and the recorded engine fixtures
  compat/rxinfer-examples/  # five RxInferExamples models, on v6 and on RxInfer's v7 branch
  investigations/           # performance investigations for the performance pass; never loaded
```

`legacy/v6/`, the unported v6 code from Phase 4.5 step 4, was deleted in Phase 6 step 10.

**Julia 1.13 only, for now** (Phase 4.5, user; `DISCUSSION.md` §3.22). Every inter-package
dependency is wired with `[sources]`, test-only ones included, via `[extras]`. The floor, and
the 1.10 workarounds described next, are revisited when the packages are registered. Until
then nothing runs in CI without a PR, and everything is verified locally.

*(What follows describes the 1.10 wiring used until Phase 4.5.)* **One cost of the 1.10 floor, found while building this:** `[sources]`, the tidy way for a
`Project.toml` to point at a sibling directory, requires Julia 1.11. On 1.10 an
inter-package dependency inside `lib/` is listed in `[deps]` — and in `[sources]`, which
1.11+ honours and 1.10 ignores — and the sibling is `Pkg.develop`ed into the environment
**at test time** (`make test-testutils`, `LibTests.yml`);
no Manifest under `lib/` is committed, so each Julia version resolves for itself. (An earlier
version of this paragraph said to commit a dev-link Manifest; one resolved on 1.10 cannot
serve 1.11 and 1.12, so Phase 4 changed it.) It was the one concrete thing the floor
decision cost. `lib/README.md` documents the current wiring: siblings in `[deps]` and
`[sources]`, test-only ones in `[extras]` and `[sources]`, `Pkg.test()` from each package's own
project, Julia 1.13, no committed Manifest.

The reason was the open items. Boundaries were still moving when this was decided — #9
through #13 were API decisions that had not landed (#9–#12 have since been resolved at the
Phase 3 sign-off, and #13 is parked) — and a change that spans two packages is one commit in a
monorepo and two pull requests plus a dev-pin across repositories. Paying the split cost
once, at a known gate, beats paying a coordination cost on every commit until then.

**What can and cannot share an environment.** The new rule packages do not depend on
ReactiveMP, and they are differently named, so `ReactiveMP@6.5.0` and
`StandardMessagePassingRules` coexist happily: the Phase 4 migration checker can call
`ReactiveMP.rule(...)` and `message_passing_rule(...)` in one process, and `MIGRATION.md`'s
before/after doctests can both execute. *(Superseded by `DISCUSSION.md` §3.36: the guide is a docs
page whose v7 side alone runs; its v6 side is shown, never executed.)* What cannot coexist is ReactiveMP v7 against v6 —
same package name. **So Phase 4.5 and Phase 7 engine comparisons must run against values
recorded from v6, not a live side-by-side**, which is why that checker must
capture results rather than only assert equality. The engine itself is rewritten in place
in `src/`, with no bridge letting v6 call the new rules; the v6 rule system and every
unported node moved to `legacy/v6/` in step 4, after their fixtures were recorded
(`DISCUSSION.md` §3.18, §3.22), and the directory was deleted in Phase 6 step 10, once every
node was ported.
ReactiveMP takes a hard `[deps]` entry on `MessagePassingRulesBase`, wired the same way.
**Downstream breakage before the release is accepted**: only a small internal group uses the
branch and checks it locally, and the coordinated downstream CI stays a Phase 8 gate.
**v7 is an evolution of the engine, not a new one.** The reactive machinery is kept: Rocket
streams, variables, the equality chain, products, scores, and deferred messages that
materialise exactly as in v6, which is load-bearing for correctness. What is replaced is how
rules are found, fetched and called, and how nodes and rules are defined and created. RxInfer
adapts where node and rule creation changes, as its own major release (`DISCUSSION.md` §3.19;
the signed-off design brief is in `PHASES.md` § Phase 4.5).

### Approximations are utilities, not algorithms

`MessagePassingRulesApproximations` is **not** part of the algorithm hierarchy and does not
depend on `MessagePassingRulesBase`. They are siblings. "How do I approximate this integral"
is a numerical utility; "which rules run" is an algorithm. The
delta node's algorithm *uses* `Unscented`; it is not `Unscented`. Delta's algorithm is a
Delta-owned value holding the approximation method and the optional known inverse, as v6's
`DeltaMeta(method, inverse)` did; case (d) built it as `DeltaApproximation(; method, inverse)`
in `lib/DeltaMessagePassingRules`. Keeping them separate
leaves the numerics usable outside this ecosystem and keeps the dependency graph flat.

Node packages depend on both where they use the numerics: Delta, Flow, Probit, GCV and Pólya.

**Measured disposition of `src/approximations/` (~1922 lines).** Deleting the unused parts
is what actually removes the heavy cubature dependencies — repackaging them would not have.

| file / symbol | fate | evidence |
|---|---|---|
| `unscented.jl`, `linearization.jl`, `approximations.jl`, `shared.jl` | → `MessagePassingRulesApproximations` | used by delta + flow |
| `rts.jl` (`smoothRTS`) | → same package | `rules/delta/unscented/marginals.jl:25`, `rules/delta/linearization/marginals.jl:27` |
| `cvi.jl` (`ProdCVI`, aliased `CVI`), `optimizers*` | **delete** | superseded — its own docstring reads *"`ProdCVI` is deprecated in favor of `CVIProjection`"* |
| `cvi_projection.jl` (`CVIProjection`, sampling strategies) | → Delta node package, implementation in its extension | see *CVI projection* below |
| `gausshermite.jl` (`ghcubature`) | → `MessagePassingRulesApproximations`, with `approximate_meancov` (§3.40) | Pólya, Probit's energy and GCV's `ExponentialLinearQuadratic` use them |
| `sphericalradial.jl` (`srcubature`) | **delete** | no consumer |
| `gausslaguerre.jl` (`glcubature`) | **delete** | no consumer |
| `importance.jl` | **delete** | no consumer |
| `laplace.jl` | **delete** | no consumer |

Dependency consequences: **`Optim` leaves ReactiveMP entirely** (only `laplace.jl` used it);
**`DiffResults` leaves with `cvi.jl`**, its only user; **`ReactiveMPOptimisersExt` and the
`Optimisers` weakdep are deleted outright** — that extension exists solely to supply
`cvi_setup!`/`cvi_update!` for the removed method. `FastGaussQuadrature` follows
`ghcubature` to `MessagePassingRulesApproximations` (§3.40); `DomainIntegrals` and `HCubature` go to
`MessagePassingRulesTestUtils` (they serve the rule-comparison quadrature in
`src/rule.jl:1464`, which is test machinery); `DomainSets` stays with
`StandardMessagePassingRules` (`normal_mean_variance/var.jl`; the ported `gamma_shape_rate/a.jl`
does not need it).

What survives is small: `Unscented`, `Linearization`, Gauss–Hermite cubature,
`approximate_meancov`, `smoothRTS` and the shared point/weight machinery, pure numerics over
means and covariances: `LinearAlgebra`, `FastCholesky`, `FastGaussQuadrature` and `ForwardDiff`
(§3.40). The methods of `approximate_meancov` that take a distribution stay with the node
packages, so the package still depends on no distribution package. *(Porting found that v6's
`unscented.jl` also used ExponentialFamily, for `JointNormal`; the port does without it.)*

**A capability regression — accepted, with conditions.** (An earlier draft called it *the
only* one. That was wrong: `srcubature`, `LaplaceApproximation` and
`ImportanceSamplingApproximation` are **exported public API**, so deleting them is a
regression too, even though nothing inside `src/` uses them. "No consumer in `src/`"
establishes absence of *internal* use, not absence of downstream use — see Open item #14.)
Alongside the exported deletions, this change needs specific migration guidance: the delta
node's built-in method set (`is_delta_node_compatible`) shrinks from
`{Unscented, Linearization, ProdCVI}` to `{Unscented, Linearization}`, so a model that runs
today on a plain `add ReactiveMP` may afterwards need a second package installed before it
runs at all. Note the two survivors are the same *kind* of method — moment propagation
through a deterministic function — while `ProdCVI` was the sampling-and-gradient one reached
for when linearization is not good enough. The capability is not lost (`CVIProjection`
supersedes it) but it moves behind an install.

**Decision: accept it, and make the diagnostic carry the weight.**

1. **Document it as a breaking change in the release notes**, called out explicitly rather
   than folded into the general list of renames — users may need an additional install as
   well as a code edit.
2. **The error must be actionable.** Using a delta node with a non-conjugate factor, or
   naming `CVIProjection` without the package loaded, must produce a message that names the
   package to install *and* the method to switch to. A `MethodError`, or a generic "no rule
   found", is a failure of this requirement. Discovery of loaded rules alone cannot supply
   this information, which is why the host guards it explicitly with the
   `is_delta_node_compatible` trait, as v6 already does (open item #11). This is a concrete
   acceptance test for the diagnostics.

### CVI projection, and a hypothesis about delta layouts

`CVIProjection` spanned awkward territory in v6: the type lived in `src/approximations/`, the
rules and a *layout* in `ReactiveMPProjectionExt`, and the layout was engine code — it
constructs `MessageMapping`, calls `connect!`, wires Rocket streams. It now lives in
`lib/DeltaMessagePassingRules/src/cvi_projection.jl`, its rules in the package's extension on
ExponentialFamilyProjection, and the layout is gone.

**Hypothesis: `AbstractDeltaNodeDependenciesLayout` is a bespoke version of the dependency
language.** Four layouts exist (default, known-inverse, CVI, CVI-projection), implementing
or delegating `deltafn_apply_layout` for the same four targets — `q_out`, `q_ins`, `m_out`,
`m_in_k`. Much of their code is repeated wiring, but they also carry stream aliasing,
static-input gating and initialization semantics. Old CVI is a migration reference, not a
surviving implementation requirement. The CVI-projection layout's own docstring reads as
a dependency spec: *"`m_in_k`: uses the inbound message on the `in_k` edge and `q_ins`"*.

*Confidence: moderate.* This is inferred from docstrings and method shape, and the layout
does make ~10 real engine calls — some may be genuine topology rather than dependency
choice (`q_out` "mirrors the posterior marginal" sounds like stream aliasing, not a rule).
**Test it in Phase 0**, where the delta layouts are the hardest thing the dependency
language would have to express. Finding its limits on the hard case is the point.

**Phase 0 tested it: the hypothesis holds for input selection, and only for that.** Written
out as declarations, the four layouts differ in exactly one respect — which messages and
marginals each of the four slots consumes. But three things in the same files are *not*
dependency choices and do not collapse: **static gating** (`with_statics`,
`default.jl:22-44`, measured as 0 emissions before a static input arrives and 2 after), the
**`N === 1` empty-group branch** (`default.jl:321-327`), and **`q_out` aliasing**
(`default.jl:47-64`), which is topology rather than a rule input. Each needs explicit support
in `MessagePassingRulesBase`, or it lands back in the engine and takes the layout with it.
See `DISCUSSION.md` §3.15.

**On that condition the plan is**: collapse layouts into dependency declarations
first *(done for the default and known-inverse layouts in Phase 4.5 case (d): two
declarations of `DeltaApproximation`; the CVI-projection layout in Phase 6)*, after which `CVIProjection` has no engine half at all — just an algorithm struct, a
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
`unscented_statistics`, `smoothRTS` — with a minimal numerical protocol replacing global
`cholinv` calls. That protocol is open item #13; it must not require a dependency on the
base package. A broader numerical API redesign remains out of scope.

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
honestly MIT. *(Done in Phase 6 step 6: only `PolyaMessagePassingRules` depends on the sampler;
its `LICENSE` file comes with registration, Phase 8.)*

**Hard constraint: `MessagePassingRulesBase` must not depend on `ExponentialFamily`.**
BayesBase exists precisely to hold this machinery. Where a needed piece is missing from
BayesBase, **add it to BayesBase**; do not reach for `ExponentialFamily`. Enforce with a CI
assertion that `ExponentialFamily` is absent from the base's dependency closure, so the
constraint survives contributors and sessions rather than eroding the first time someone
wants one convenience function.

**Corollary: those BayesBase additions must ship as non-breaking 1.x releases.** The
v6/v7 comparison harness (see Repository layout) puts `ReactiveMP@6.5.0` and the new rule
packages in one environment, which only resolves because v6.5.0 declares `BayesBase = "1.5"`
and `ExponentialFamily = "2.5.0"` — caret bounds, so new *minor* versions are fine. A
BayesBase 2.0 released for this rewrite would make that environment unresolvable and
silently cost us the migration checker, which is the main instrument for verifying 490
ported rules. If a breaking BayesBase change becomes unavoidable, the comparison harness
needs redesigning first, not afterwards.

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

Follow `PHASES.md` for the authoritative order and acceptance criteria: preparation and
baselines → the throwaway dispatch/dependency spike → external feedback and tooling → base
and test utilities → **Phase 4.5: the engine design session and the first cut of the real v7
engine, before bulk migration** → standard rules, ported straight into it → approximations
and node packages → completing the engine → **a cleanup that rids the repository of the
rewrite's historical remarks** (phase and step references, what v6 did, pointers into these
documents, and these documents themselves; git keeps the history, `PHASES.md` § Phase C) →
coordinated release. There is no bridge into
the v6 engine; v6 is a source of recorded fixtures only (`DISCUSSION.md` §3.18).
The tooling work retains TestItemRunner and adds name/tag filtering, Runic and Aqua checks.

This work spans multiple sessions and wants external feedback. Carry it on a long-lived
branch with a repo-level `PLAN.md`; circulate the spike results before building the macros.
The dispatch result, ownership contracts and early engine integration are separate gates.

## Files

*(Rewritten at the Phase 4.5 reconciliation for the clean cut, §3.22.)*

- `src/rule.jl` (1984 lines), `src/rules/`, `src/nodes/predefined/`, `src/approximations/`,
  `ext/` — **moved to `legacy/v6/` in Phase 4.5 step 4**, never loaded; Phases 5 and 6 ported
  them into `lib/`, node by node, and Phase 6 step 10 deleted the directory. v6's code is in
  the 6.5.0 release and in git.
- `src/nodes/nodes.jl` — `@node` and the v6 traits are gone; `FactorNode`,
  `factornode` and `activate!` stay in the engine and are rebuilt on the `NodeSpec`:
  `(name, index)` interfaces, clusters as interface-name tuples, arity and aliases from the
  spec, generic activation with no per-node override.
- `src/nodes/dependencies.jl`, `src/nodes/clusters.jl` — the default scheme stays; the
  per-algorithm `dependencies_spec` replaces `functional_dependencies`; a local marginal
  carries its member tuple, never a joined name.
- The mixtures' per-node `factornode`/`activate!`/`collect_latest_*` are not ported: groups and
  declared dependencies replace them (brief item 2).
- `src/nodes/predefined/delta/` — in case (d) its layouts became a Delta-owned
  algorithm, `DeltaApproximation` in `lib/DeltaMessagePassingRules`, and engine features keyed
  off the spec: static folding and gating (`src/nodes/static_inputs.jl`), `q_out` aliasing and
  the empty group.
- `src/nodes/interfaces.jl` — v6's `ManyOf` and its helpers are deleted (case (c)); a group
  reaches a rule as a tuple.
- `src/message.jl`, `src/marginal.jl` — stay in the engine, value types included:
  `Message{D, L}` and `Marginal{D, L}`, `L` the log scale (`DISCUSSION.md` §3.50). Rules see
  raw distributions in `args`, the incoming log scales in `args.logscale` (for a rule declaring
  `reads_logscale = true`), annotations in `ann` and the node in `ctx.node`, so the envelope
  (`is_clamped`, `is_initial`, annotations, log scale) is unwrapped by the engine before a rule
  is called and the base package never knows it (decided while planning Phase 3).
- `src/scratch.jl` — the scratch slot each message and marginal mapping keeps (§3.44);
  `src/diagnostics.jl` — `EngineDiagnostics` (§3.46); `src/context.jl` — the node's
  `RuleContext` (§3.49).
- `src/nodes/equality.jl` — `BitVector` caches → `Vector{Bool}` (bit-packed writes are
  read-modify-write on a shared word; neighbouring indices race).

## Open items

1. ~~**Rule syntax final form.**~~ **RESOLVED.** The surface is fully keyword-based with an
   ordinary lambda body over a real arguments object, symbols throughout
   (`target = :out`, `m[:μ]`, `interfaces = [:out, ...]`), group members as `q[:p][k]`,
   indexed targets as `(:m, k)`, body slots `(output, scratch, algo, ctx, args, ann)`
   (`scratch` added by §3.44; a slot `node` dropped by #12) in
   canonical order, and dispatch carried by the `algorithm` keyword. `@allocate` and
   `@logscale` are deleted rather than renamed, into the `preallocate` and `logscale`
   keywords. See § Rule surface.
2. **`aligned` generality** — everything in-tree is `k ↔ k`. `q[:p][f(k)]` extends
   naturally; don't build until something needs it. (Not `q[:p[f(k)]]`, which parses as
   `(:p)[f(k)]` — indexing a `Symbol`. See § Rule surface.)
3. ~~**Per-(target, factorisation) group selection.**~~ **RESOLVED at the Phase 3
   sign-off: per target.** No in-tree rule chooses group members from the factorisation —
   the mixtures reject anything but mean-field, and `Mixture` ignores the factorisation. Since
   dependencies belong to the algorithm, selection that genuinely varies with factorisation is
   written as a distinct algorithm. The door stays open without a syntax axis.
   *(Refined at the Phase 4.5 algorithm reconciliation: under `DefaultAlgorithm` the engine's
   default scheme already follows the factorisation, so no algorithm is needed for that. A
   node that **ignores** the factorisation, as the mixtures do, declares its own algorithm,
   e.g. `NormalMixtureVMP`.)*
4. **Ruleset axis** (`StandardRules()`, `Overlay(mine, standard)`). **DEFERRED in Phase 0.**
   A downstream package that wants its own rule for a standard node and edge declares its own
   algorithm and gets it, with no shadowing and no ambiguity, because the algorithm is part of
   the signature. The piracy argument for the axis was already dead for rules (see § Testing;
   node declarations are flagged and declared owned). Adding
   the axis later is a new keyword rather than a resurfacing, so it waits for a concrete use
   case. *(Since the Phase 4.5 algorithm reconciliation a `DefaultAlgorithmExtension` gives a
   one-level overlay without any ruleset keyword: its own rules first, the default's for the
   rest. The general `Overlay(mine, standard)` stays deferred. If a use case needs an overlay
   over a node's own algorithm, the recommended form is an extension with an explicit parent,
   `AlgorithmExtension{Parent}`, rather than a registry axis; both are sketched in
   `DISCUSSION.md` §3.23.)* The rule-fallback contract was
   specified independently, as required:
   **resolution is a separate, total function** — `find_rule` returns a spec or a
   `RuleNotFound`, never throws and never runs anything, and the fallback is consulted on the
   not-found branch only, which is decided before any body runs. An exception from inside a
   selected rule therefore cannot reach the fallback, structurally rather than by discipline.
   This removes v6's asymmetry, where `rule` returns a sentinel and `marginalrule` throws.
   Built as the activation option `rulefallback`, with the base's `NodeFunctionRuleFallback`,
   v6's computation from the node function (`lib/MessagePassingRulesBase/src/fallback.jl`;
   `DISCUSSION.md` §3.49).
5. **Reactant and StableCholesky** — deferred to their own effort. Per-rule compilation is
   an explicitly supported *research* path when it happens, not a rejected one; whole-sweep
   tracing and `vmap` are a superset of it, not a competing approach. Nothing to decide
   here beyond keeping `buffer_like` extensible.
6. ~~**`reverse(...)` in the mixture marginal wiring**~~ **RESOLVED in Phase 4.5 case (c), by
   not reproducing it** (user; `DISCUSSION.md` §3.24). The engine subscribes in declaration
   order, which is the VMP update schedule; the group order was found to change the
   trajectory (the same optimum, slower), so `NormalMixture` declares its precisions first,
   and the member reversal, which changes no value, is dropped. The original entry, whose
   "inert" is half wrong (Correction 26): undocumented, and the two groups are
   additionally swapped relative to the payload. Located: it is **not** in `mixture.jl` but in
   `normal_mixture.jl:176,183` and `gamma_mixture.jl:166,173`. Measured to be
   **observationally inert**: the `reverse` applies only to the `combineLatest` *trigger*
   tuple, while the emitted payload comes from `map_to` with the groups un-reversed, and
   `combineLatest` gates on the *set* of streams rather than their order. So the regression to
   pin is **scheduling, not values** — cheaper than this entry originally implied, but still
   pin it before touching, because bit-identical scheduling is what Phase 4.5 compares.
   Since the v6 wiring is replaced rather than bridged, "pinning" means recording v6's
   emission order as a fixture (Phase 4.5 Step 0) that the new engine must reproduce.
7. ~~**`EdgeLabel.index`**~~ **RESOLVED** (engine side in Phase 4.5 case (c), RxInfer side in Phase 7 item 6). It exists in GraphPPL but RxInfer discarded it; ReactiveMP re-derived
   group indices from position, silently depending on neighbour order. Plumb it through.
   *(`factornode` takes `((:m, k), variable)` and the index reaches the rule's `k`, verified
   by case (c); RxInfer's `refactor/reactivemp-v7` branch passes `EdgeLabel.index` as `k`.)*
8. ~~**Does `MessagePassingApproximations` exist at all?**~~ **RESOLVED.** Yes, as
   `MessagePassingRulesApproximations`, holding `Unscented`/`Linearization`/`smoothRTS`,
   Gauss–Hermite cubature and `approximate_meancov` (§3.40), and shared point/weight machinery — **standalone numerical utilities that do not depend
   on the base package**. Old CVI and unused methods are deleted; CVI projection belongs to
   the Delta package and its extension. See § Approximations are utilities, not algorithms.
9. ~~**Beliefs consumed vs. the partition whose entropy is counted.**~~ An earlier draft
   required all requested marginals to form the entropy partition. That conflated two
   things: a rule may consume an *auxiliary* belief that is
   not an entropy cluster — `RequireMarginalFunctionalDependencies` already does exactly
   this. Define separately: which beliefs a rule consumes, and which partition free energy
   is computed over. Also settle auxiliary marginals, unspecified interfaces, conflicting
   user-supplied factorisation, and **whether `q[:a, :b]` and `q[:b, :a]` are distinct ordered
   inputs or require permuting the joint** — canonicalising the names is not sufficient.
   **Decide before the macro surface freezes.**
   **RESOLVED at the Phase 3 sign-off.** Two separate declarations: **consumed** (a
   dependency's right-hand side) and **partition** (derived from the factorisation, or
   declared by the algorithm). An auxiliary marginal is consumed and never scored — as v6's
   `RequireMarginal` and `ContinuousTransition(:a)` already behave. A joint lists its members in
   interface-declaration order (`q[:y, :x]` when `y` is declared first); `check_rules()` rejects
   any other order at definition time and nothing is ever permuted. Every in-tree joint already
   follows that order. A user factorisation that conflicts with an algorithm's declared
   partition is an activation-time error naming the algorithm.

10. ~~**Buffer ownership, as distinct from buffer allocation.**~~ `preallocate` answers *how to
    create* storage, not *when it may be reused*. Retainers beyond the equality chain:
    `DeferredMessage` caches its result, subjects retain recent messages, and
    `InputArgumentsAnnotations` stores references to inputs *and* results — so recording a
    rule, running another iteration into the same storage, then inspecting the record shows
    the new values. Specify: whether published results are snapshots or borrowed; when reuse
    becomes legal; how retained annotations, callbacks and subscribers affect eligibility;
    and behaviour when dimension or element type changes. Note **poisoning is not
    sufficient** — a stale reference can read a legitimately rewritten buffer and see
    plausible but wrong values. Test retention across multiple updates.
    **RESOLVED at the Phase 3 sign-off.** Buffers and published messages are **engine
    internals**. Whether the engine reuses the storage behind a message, and when, is
    decided by the engine alone and is **deliberately unspecified** — it may or may not
    happen, and may change between releases. Anything outside the engine (callbacks,
    subscribers, user code) must copy what it wants to keep; any getter the engine offers to
    outsiders **copies by default**. `InputArgumentsAnnotations` records **references** to its
    inputs and result, not copies (`src/annotations/input_arguments.jl`): safe only because
    every rule output is freshly allocated, and load-bearing once output buffers are reused
    (tracked in `PHASES.md`'s not-done table). Phase 3's base package defines only `preallocate`/`rule!`; the
    engine-internal retainers (`DeferredMessage` cache, equality-chain caches, subjects) are
    the engine's own eligibility problem, settled in Phases 4.5/7.

11. ~~**The missing-capability diagnostic needs metadata beyond loaded rules.**~~ A registry
    that only discovers *loaded* modules cannot know which *unloaded* package supplies a
    missing capability — yet the CVI regression above is accepted on the condition that the error
    names the package to install. Needs an explicit capability declaration available in
    the already-loaded host (such as Delta), or a static table. Its representation remains
    open; placing it only in the unloaded extension would not solve the problem.
    **RESOLVED at the Phase 3 sign-off: no new mechanism.** v6 already solves this with an
    explicit guard, and the design keeps it. The host (Delta) defines a trait
    `is_delta_node_compatible(method)`, `Val(false)` by default, and checks it when the
    method is attached (`DeltaMeta(; method)`, `delta.jl:19-32`). A method the host knows
    about but whose implementation lives in an extension gets a specialised error in the host
    — `cvi_projection.jl:138-140` names `ExponentialFamilyProjection` — and the extension
    flips the trait to `Val(true)` (`ReactiveMPProjectionExt.jl:64`). The static capability
    table proposed in the entry brief was rejected as redundant. Two small follow-ups when it
    moves: the error must also name the method to switch to (today it names only the
    package), and the check belongs in an inner constructor, since the positional
    `DeltaMeta{M, I}(…)` bypasses it (nothing in-tree calls it that way).
    *(Case (d) moved the guard to `DeltaApproximation`, and Phase 6 step 2 settled both
    follow-ups: the check is in its inner constructor, so the positional form cannot bypass
    it, and the error names the methods the node takes and the package that supplies
    `CVIProjection` (`lib/DeltaMessagePassingRules/src/node.jl:41–66`).)*

12. ~~**Context service contracts.**~~ Phase 0 turned both hard cases into signatures, each
    demonstrated as a standalone call with no graph and no Rocket:
    `product : (left, right) -> (dist, logscale::Real)`, the log scale being the product's own
    and not the inputs' (Phase 5 step 8, `DISCUSSION.md` §3.34; the service is removed,
    §3.50), and
    `nodefn : (ctx, target) -> a callable of the free arguments only`. Neither carries an
    engine type. **Still open: incoming annotations have no declared route.** The switch rule
    needs the log scales that *arrived* with its messages, but `args` holds message data and
    `ann` is an output sink the rule writes to. Recommended: a parallel accessor keyed exactly
    like `m` — `args.ann_in[:out]` — kept out of dispatch, because an annotation must never
    select the mathematics. Settle the representation before the macro surface freezes. Also clarify that "context is non-dispatching" means it does not select
    the mathematical rule — its concrete services may still specialise for efficiency.
    **RESOLVED at the Phase 3 sign-off.**
    - Incoming annotations go through the existing `ann` slot, which carries both
      directions: read `ann.m[:out]` / `ann.q[:μ]`, write `annotate!(ann, …)`. The separate
      `args.ann_in` accessor is not adopted. See § Rule surface. Incoming **log scales** are
      not annotations: a rule declaring `reads_logscale = true` reads them as
      `args.logscale.m[:x]` (`DISCUSSION.md` §3.50).
    - The context holds a reference to the node, `ctx.node`, so the `node` body slot is
      dropped: slots are `(output, scratch, algo, ctx, args, ann)`. Nothing dispatches on the node.
    - `nodefn` is therefore not a service: a delta rule calls `getnodefn(ctx.node, …)`.
    - The `product` service is removed (§3.50): Mixture's switch rule, its one user, computes
      the product's log scale under `MixtureBP(; prod = GenericProd())`.
    - The engine's default services are `node`, `rng` and `matrix_correction`, a
      MatrixCorrectionTools strategy (Phase 5 step 5, user); `nothing` means not set: a rule
      reads it through `matrix_correction(ctx, default)`, falling back to its own default, and
      an explicit identity is `NoCorrection()` (§3.31). Any other name is allowed, supplied
      through the activation option `context` (§3.49). `RuleContext{S <: NamedTuple}` is
      parameterised, so services specialise. There is no `linalg` service.
    - **Missing inputs behave exactly as in v6** (user, closing Phase 3): when any input is
      `missing`, the rule body and the post-rule annotation processors are both skipped and
      the result is `missing`. Written into the `execute_rule` docstring; pinned by an engine
      test when the engine is rewritten.

13. **The approximation package's numerical protocol is unspecified.** Passing a context
    value need not itself introduce a package dependency, but requiring Base-owned types
    or services would violate the boundary. Define the minimal protocol it accepts, for
    example a factorisation strategy and workspace, without depending on
    `MessagePassingRulesBase`. The representation remains open until Phase 3.
    **Parked by the user until the late phases** — no proposal is on the table. The entry
    brief's `approx_cholinv`/`approx_cholsqrt` idea was set aside for further thought, not
    rejected; do not treat it as the plan. What is parked is the rules' and the numerics'
    direct FastCholesky calls; no context service stands in for it.

14. ~~**A complete disposition inventory is missing.**~~ **RESOLVED in Phase P.**
    `INVENTORY.md` assigns a destination or a deliberate deletion to all **231** entities —
    49 nodes, 165 exported symbols, 8 engine-hook families, 2 extensions and 7 rule-level
    exceptions — including the five hook families that export nothing yet are documented
    public API (callbacks, stream postprocessors, delta layouts, the CVI optimiser hooks,
    and the `@node`-generated traits). Rules inherit their node's destination; only the
    ones that cannot are listed individually. All 18 exported deletions (21 deletion rows at the time; 22 since `NormalMixtureNode` joined them in Phase 4.5) carry a
    migration note, including where the answer is "no replacement".

    Generated and validated by `scripts/inventory.jl`, gated by the root suite's
    `test/inventory_tests.jl` (tag `:quality`), so a node added upstream without a
    destination fails the suite rather than being silently missed at split time.

    Two findings came out of building it, both recorded in the inventory notes:
    `CompanionMatrix`/`CompanionMatrixTransposed` have **no reference in `src/` or
    `test/`** — the autoregressive node uses a companion-matrix representation but through
    its own `ARTransitionMatrix` (`autoregressive.jl:270`), which superseded them *(wrong: AR
    builds them with `as_companion_matrix`, and they go to its package; Phase 6 entry brief)* — and the
    three `src/helpers/algebra/` files between them account for **253 of the 322** Aqua
    ambiguities, so that cleanup is entirely separable from the rewrite.

## Migration

*(Revised at the Phase 5 entry brief, user: **no transform tool**. The rules are ported by hand
or by agent, per directory, each gated by v6's own tables, a `compare_standard.jl` case per
rule, verification against the node definition where it applies, and `check_rules`. The first
three bullets below are the tool's plan, kept for the record; the `y_x` question they raise is
answered per rule by reading the v6 node's interfaces. See `DISCUSSION.md` §3.26 and
`PHASES.md` § Phase 5.)*

- 490 rule definitions (384 + 106) + 172 `@call_rule`/`@test_rules` sites in tests. Build the transform on
  **JuliaSyntax** (source-preserving green tree), not regex and not MacroTools — bodies
  contain arbitrary code, `where` clauses and comments worth keeping.
- Run the tool with ReactiveMP v6 loaded so it can call `interfaces(fform)` as an
  **oracle** to decide whether `y_x` is one interface or the cluster `(:y, :x)`. Without it
  you are regex-guessing on `_`, which is the exact bug class being deleted.
- Migrate per rule directory (50 of them), reviewing diffs directory-by-directory.
  `@test_rules` gives per-rule numerical regression coverage for free.
- A rule that calls another rule (`@call_rule` in a body) calls a helper function holding the
  shared mathematics instead (user, Phase 5).
- Hand-written: `mixture/switch.jl`, the ~15 rules touching raw `messages[i]`/`marginals[i]`
  tuples, the 5 `MessageMapping` construction sites and 4 delta layout files. *(Counted at the
  Phase 5 brief: three of the raw-indexing rules are in `standard`, the `Mixture` rules; the
  rest are DiscreteTransition's. The delta layouts were done in Phase 4.5 case (d).)*
- Canary: `NormalMixture((:m, k))` — indexed target + `ManyOf` marginals + `where {N}` +
  aligned group dependency in one rule. *(Ported in Phase 4.5 step 3.)*

## Migration guide (published, for downstream authors)

Distinct from the internal transform above, which was dropped (Phase 5 entry brief), the rules
being ported by hand or by agent instead; **this is a durable, published document** for anyone maintaining their own nodes and
rules — RxGP, and colleagues with custom rules in their own codebases. It must work for a
human reading it *and* for an AI agent pointed at it, since that is how much of the
downstream migration will actually happen.

Lives as a page of ReactiveMP's docs, `docs/src/migration-guides/v6-to-v7.md`, next to the
v5 → v6 guide, and is linked from RxInfer (user, Phase 5 step 9, `DISCUSSION.md` §3.36; there is
no separate `MIGRATION.md`). Only its v7 code runs: the v6 side of each pair is shown for the
reader and never executed, since v6 leaves the repository with the refactor.

**Requirements that make it usable by an agent, not just readable:**

- **Mechanical before/after pairs, not prose.** Every v6 construct maps to its v7 form as a
  concrete pair. An agent should not have to infer the rule from a description.
- **Complete case coverage**, including the fiddly ones — `ManyOf` → variadic groups,
  indexed edges `(:in, k)`, joint marginals `q_y_x` → `q[:y, :x]`, `meta` → `algorithm`,
  `@logscale` becoming the `logscale` keyword (§3.50), `getnode`/`getnodefn`, `Marginalisation`
  removal, the renamed macros and the move to the keyword form.
- **An explicit "cannot be translated mechanically" section.** Rules touching raw
  `messages[i]`/`marginals[i]` tuples, rules constructing graph objects, anything relying on
  `meta` as a mutable workspace (the RxGP `GPCache` pattern). An agent must be told to stop
  and ask rather than guess — silently-wrong rules are the worst outcome here.
- **A verification procedure, not just a rewrite procedure.** Ship a checker in
  `MessagePassingRulesTestUtils` that runs the v6 and v7 rule on identical inputs and
  asserts agreement. This is what makes agent-driven migration trustworthy rather than
  hopeful: an agent that can check its own work is a different proposition from one that can
  only pattern-match. Node-definition verification is the stronger form where it applies.
- **Executable examples.** The v7 side of every pair is a doctest that the docs build runs,
  so the guide cannot drift from the code. The v6 side is shown as plain code: it is not run,
  in CI or elsewhere (user, Phase 5 step 9).
- **A short preamble addressed to an agent** — what to read first, what to never guess at,
  how to verify, when to stop and ask.

**Write it during the migration, not after.** The mechanical rules get discovered while
porting our own rules; reconstructing them later from memory guarantees the guide is
incomplete in exactly the places that were fiddly. **The tool and the guide should be
derived from one source** — if the transform encodes a rule, the guide documents that same
rule, with a test asserting they agree. *(Dropped with the tool at the Phase 5 entry brief:
the guide is written from what the hand ports find, its pairs as doctests.)*

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
  Tag taxonomy of the root suite: `:nodes`, `:engine`, `:alloc`, `:quality`, `:slow`; rules
  are tested in the lib suites. `make test` = fast subset / `make test-all` = everything. Precompilation, not execution, is
  the real latency cost in an agent loop.
  **Revisit ReTestItems only if CI wall-clock becomes the bottleneck** — its one real
  advantage is distributed parallel workers, a CI argument rather than an iteration-speed one.
- **Runic** replaces JuliaFormatter — **done in Phase 2**. Deterministic and zero-config,
  which eliminates by design the formatter-version drift CI and contributors used to hit;
  measured byte-identical on 1.10 and 1.13. Already in use in StableCholesky.jl.
- **Aqua checks — done in Phase 2, `ambiguities` since.** `piracies`, `deps_compat`'s
  `check_extras` and `ambiguities` are on in `test/runtests.jl`, and each lib suite checks its
  own package. `ambiguities` was off in Phase 2: 322 pairs, of which 253 came from
  `src/helpers/algebra/` and 27 from rule dispatch, both gone with v6's code. The reasoning
  below is what led there.
  - `ambiguities`: budget it separately. The key-set argument (rules with
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
  - **Caveat, also measured: the piracy check is vacuous for *message* rules, and always will
    be.** *(Phase 4.5 found it is not vacuous for a rule package as a whole: `nodespec`,
    `nodefunction`, average energies and marginal rules for another package's distribution
    are flagged, so `StandardMessagePassingRules` declares its node types owned with
    `treat_as_own`. Only message rules escape, through their target's `Symbol`.)*
    Aqua's rule is that a `DataType` is foreign only if the type *and every one of its
    parameters* is foreign, and `is_foreign(::Symbol) = false` unconditionally. Verified:
    `is_foreign(Val{:out}, pkg) == false` while `is_foreign(Val{1}, pkg) == true`. Since
    every rule target carries the edge name as a `Symbol` type parameter — `Val{:out}`
    today, `MessageTarget{F, :out, …}` in the new design — no rule can ever be flagged,
    whoever defines it. So enabling the check is worth doing, but **it must not be claimed
    as evidence that the rule-package split is piracy-clean**; it says nothing either way.
    This also removes piracy as an argument for the ruleset axis (open item 4).
- **Expand JET well beyond its current two uses.** It enforces the devirtualization gate
  for the new routing machinery. Rule bodies and user-supplied services are assessed
  separately; the routing assertion catches regressions in the core design.

### The `@test_rules` successor

Lives in **its own package, `MessagePassingRulesTestUtils`**, not in `src/` as it does
today. Consumers — `StandardMessagePassingRules`, a hypothetical `AutoregressiveNode` —
list it under `[extras]` and the `test` target, with `[sources]` while it is unregistered, so its dependencies (quadrature, sampling,
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

The machinery existed in v6: `@node` generated `nodefunction` (the node's logpdf);
`src/approximations/` had `ghcubature`, `srcubature` and importance sampling; `rules/fallbacks.jl`
built an unnormalised logpdf from the node function. Here `@define_factor_node` declares
`nodefunction`, `MessagePassingRulesTestUtils` holds the verification, and
`NodeFunctionRuleFallback` builds the logpdf (§3.49). So the
reference update can be computed numerically from the node definition — BP as
`∫ f(x) ∏_{j≠i} m_j dx_{≠i}`, naive VMP as `exp(E_{q(¬i)}[log f])` — and compared to the
analytic rule. This tests the *maths*, not a regression table, and is what makes porting
490 rule definitions credible.

A sampling-based variant (e.g. building the local factor as a Turing model) is an **idea,
opt-in behind its own flag**, not a default. It has one subtlety that must be deliberate or
the test silently measures the wrong thing: a BP message is not a posterior. Sampling the
local factor yields a *marginal*, whereas the message is that integral with no prior on the
target edge. So the comparison must be: rule message × a known proper test prior on the
target edge, against the MCMC marginal obtained under that same prior. Done naively it
appears to work on symmetric conjugate cases and fails confusingly elsewhere.

**The test-prior trick validates shape, not scale.** Multiplying by a proper prior and
comparing against the MCMC/quadrature marginal checks the *normalised* posterior shape — it
will happily pass a message whose normalisation constant is wrong. Since log scales feed
free energy and the `Mixture` rules, test distribution shape and normalisation constant as
**separate assertions**.

**Define a bounded minimum subset before building it**, or this becomes an open-ended
numerical project: low-dimensional stochastic nodes, exact enumeration for discrete cases,
controlled quadrature elsewhere, explicit normalisation handling. Grow from there.

**Define the reconciliation procedure when the oracle disagrees with v6.** "Preserve v6
output" and "fix old mathematical errors" are contradictory instructions. Required
response: investigate independently, then record the outcome as either a migration bug or a
deliberate correction with its reasoning — never silently adopt either side.

Scope honestly: stochastic nodes only (`nodefunction` is not generated for deterministic
ones); MC error forces loose tolerances and quadrature hits dimensionality fast; rules that
are themselves approximations (delta, projection) have no exact reference; improper or
unnormalised messages, `PointMass` inputs, and rules returning `ProductOf`/`FactorizedJoint`
need special handling. A property test complementing the tables, tagged `:slow`.

### Strict TDD

- **Failing test first in every PR**, unless explicitly justified in the PR why it is not
  possible or not required.
- **Registry-backed coverage, mechanically enforced:** because the definition macros emit data,
  "every rule has a test" becomes a CI check — cross-reference the registry against tested
  rules and fail on any `RuleSpec`/`NodeSpec` with no test entry. Nothing in the current
  system can do this, since rules exist only as methods.
  **Coverage must record the rule actually selected**, not merely that a test mentions some
  node and edge. Otherwise a broad fallback satisfies the check while the specialised rule
  it was meant to cover never executes. *(Enforced since the Phase 5 post-close review: the
  Standard and Delta suites run `check_rule_coverage` after an unfiltered run, and a direct
  `call_*` records its rule as a table case does, `DISCUSSION.md` §3.39.)*
- Coverage floor, fail on decrease. *(The gate requires zero gaps, which is the floor.)*

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

- Numerical regression per rule, with `@test_message_update_rule` (the `@test_rules`
  successor) and the v6 comparison, run per directory during migration.
- **Go/no-go gate before bulk rule migration:** use hand-written rules in Phase 0 to check
  with `@code_typed`/JET that the new routing machinery devirtualizes — specifically
  **through the `RuleSpec`**, including the in-place branch the spec owns, since a spec whose
  body lives in a `::Function` field would pass a naive dispatch check and still allocate on
  every message (see § Registry for the measurement). Require equivalent
  dispatch behavior and no material measured routing overhead versus direct calls, not
  identical generated code. Measure cold and warm execution, allocations and specialization
  growth; assess rule bodies and user-supplied services separately.
- `check_rules()` + `check_rule_ambiguities()` + registry/method-table consistency in CI.
- Allocation regressions, where a rule opts into the non-allocating flag: kernel (`== 0`),
  rule with a provided buffer (`== 0`), full sweep (golden number + tolerance) — three
  distinct levels, never conflated, and none of them implied by `inplace`. The root suite's
  `:alloc` items assert allocation counts; v6's precedents were `test/annotations_tests.jl:107`
  and `test/rules/mv_normal_mean_scale_precision/out_tests.jl:129-136`.
- Buffer lifetime checks: test retained results across multiple updates; `checked_buffers`
  poisons the scratch, the only memory recycled, and no CI job runs in checked mode, since the
  engine has no global switch for it. Poisoning alone does not prove safe reuse.
- Mixture rewrite: pin current behaviour (including the `reverse` quirk) first, then delete.
  *(Done by recording v6's emission order as the `normal_mixture` fixture, Phase 4.5 step 0.)*
- Log scales are first-class (`DISCUSSION.md` §3.50, superseding the Phase 4.5 scope that kept
  v6's behaviour, gaps included, and §3.48): a rule declares one where it is known, checked by
  enumeration or quadrature, and leaves it undeclared, an `UndefinedLogScale`, where unsure.
- **Annotation and product behaviour is its own acceptance gate, not a by-product of
  numerical rule tests.** Folding products over raw distributions also touches form
  constraints, fold order, callbacks, and the `is_clamped`/`is_initial` flags — and
  `Message ==` deliberately ignores annotations and log scales (`src/message.jl`), so
  numerical equality can pass while log scale bookkeeping is silently wrong. Compare
  observable behaviour explicitly: `getannotations`, `getlogscale`, flags, callback order
  where contractual, and free-energy results. Include the
  missing-input path, where today a missing input bypasses rule execution *and* the
  post-rule annotation processors.
- **A dedicated rewrite integration environment.** `.github/workflows/IntegrationTest.yml`
  catches `Pkg.Resolve.ResolverError` and calls `exit(0)`, treating resolution failure as an
  intentional breaking change. Reasonable for ordinary releases; **useless as the gate for a
  coordinated rewrite**, since incompatible versions would report green without running a
  single downstream test. Add a job pinning mutually compatible revisions of the new
  packages and their consumers, in which resolution failure is a hard failure. Start it as
  soon as compatible development revisions exist; Phase 8 requires it to pass for release.
- End-to-end: RxInfer's test suite and RxInferExamples against the new packages.
