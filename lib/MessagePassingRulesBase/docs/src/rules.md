```@meta
CurrentModule = MessagePassingRulesBase
```

# Defining rules

A node computes three kinds of thing, each defined with its own macro:

- a **message** towards one of its interfaces, [`@define_message_update_rule`](@ref);
- the joint **marginal** of a structural cluster of its interfaces,
  [`@define_marginal_update_rule`](@ref);
- its **average energy**, its term of the Bethe free energy, [`@define_average_energy`](@ref).

Each definition names its node, its target (none for an energy), the inputs it takes with their
types, and its body, an ordinary lambda. It becomes a method of [`find_message_rule`](@ref),
[`find_marginal_rule`](@ref) or [`find_average_energy`](@ref), so a rule defined in any loaded
package is found by dispatch on the node, the target, the algorithm and the inputs' types. A
rule that omits `algorithm` belongs to its node's default algorithm, so the node must be
declared before the rule is loaded.

```@docs
@define_message_update_rule
@define_marginal_update_rule
@define_average_energy
```

## Targets

A message rule's target is a single interface, `target = :out`, or any member of a group,
`target = (:in, k)`, which binds `k` to the member's index in the body and lets the inputs select
by it. A marginal rule's target is a cluster, `target = (:out, :μ)`, its members in interface
order, a group's member written `(:T, 1)`. Internally a target is a type, so rules dispatch on it:

```@docs
MessagePassingRulesBase.Target
MessagePassingRulesBase.IndexedTarget
MessagePassingRulesBase.ClusterTarget
MessagePassingRulesBase.target_edge
MessagePassingRulesBase.target_index
MessagePassingRulesBase.cluster_members
```

## Inputs

`args` lists what a rule consumes: messages `m[:x]`, marginals `q[:x]`, joint marginals
`q[:y, :x]`, whole groups `m[:in...]`, and for a group target the aligned member `m[:in][k]` or
every other member `m[:in][!k]`. A group arrives as a tuple in member order, with `nothing` where
the selection leaves a member out, so `args.m[:in][k]` is member `k` whatever was selected:

```jldoctest rules
julia> using MessagePassingRulesBase

julia> struct Sum end   # out = in₁ + in₂ + …

julia> @define_factor_node(node = Sum, type = Deterministic, interfaces = [:out, :in...])

julia> @define_message_update_rule(
           node = Sum, target = :out, args = (m[:in...]::Real,),
           body = (args) -> sum(args.m[:in]),
       )

julia> @define_message_update_rule(
           node = Sum, target = (:in, k), args = (m[:out]::Real, m[:in][!k]::Real),
           body = (args) -> args.m[:out] - sum(x for x in args.m[:in] if x !== nothing),
       )

julia> getresult(@call_message_update_rule(node = Sum, target = :out, m = (in = (1.0, 2.0, 3.0),)))
6.0

julia> getresult(@call_message_update_rule(node = Sum, target = (:in, 2), m = (out = 6.0, in = (1.0, nothing, 3.0))))
2.0
```

The body receives them as a [`RuleArgs`](@ref): `args.m`, a [`Messages`](@ref), and `args.q`, a
[`Marginals`](@ref). Which inputs a rule receives is not the rule's choice but the engine's: under
the default algorithm it follows from the factorisation (see
[Algorithms and dependencies](@ref)). A rule is defined for the inputs it will be given.

```@docs
MessagePassingRulesBase.RuleArgs
MessagePassingRulesBase.Messages
MessagePassingRulesBase.Marginals
MessagePassingRulesBase.canonical_cluster_keys
```

### Rules over whatever the factorisation delivers

A node whose rules are one computation over any factorisation, such as a tensor node, declares
`default` among a rule's `args`: the rule takes whatever inputs the default scheme delivers, and
requires the typed inputs named beside it.

```julia
@define_message_update_rule(
    node = DiscreteTransition, target = (:T, k), args = (default, q[:a]::DirichletCollection),
    body = (args) -> contract(args, k),
)
```

The body walks its inputs with [`rule_inputs`](@ref), `key => value` pairs: an interface by its
name, a group's member as `(:T, k)`, and a joint by its key. A marginal rule over any cluster
names its target with a bare name, `target = members`, bound in the body to the cluster's key. A
rule with explicit inputs for the same node and target is more specific and wins where it
applies, so a fast path can sit on top. Missing or mistyped typed inputs make the lookup a
[`RuleNotFound`](@ref). There is at most one `default` rule per node, target and algorithm.

```@docs
MessagePassingRulesBase.rule_inputs
```

## The body

The body is a lambda over some of the slots `(output, scratch, algo, ctx, args, ann)`, named in
that order, and only those it uses:

- `output`: the buffer an in-place rule writes into;
- `scratch`: its working memory;
- `algo`: the algorithm value it runs under, and so its parameters (see
  [Algorithms and dependencies](@ref));
- `ctx`: the [`RuleContext`](@ref), the services the rule declares with `ctx = (...)` (see
  [The rule context](@ref));
- `args`: the inputs;
- `ann`: the annotations.

A rule that reuses another's computation calls a plain helper function both share, rather than
the other rule.

## In-place rules

A rule that writes its result into a buffer declares `inplace = true` and how to build the
buffer, `preallocate`, over the slots `(algo, ctx, args)`; its body takes `output` first and
returns it. An engine may keep the buffer between calls. [`buffer_like`](@ref) builds storage of
the right kind from an input.

```jldoctest rules
julia> struct Double end

julia> @define_factor_node(node = Double, type = Deterministic, interfaces = [:out, :in])

julia> @define_message_update_rule(
           node = Double, target = :out, args = (m[:in]::Vector{Float64},),
           inplace = true,
           preallocate = (args) -> MessagePassingRulesBase.buffer_like(args.m[:in]),
           body = (output, args) -> (output .= 2 .* args.m[:in]),
       )

julia> getresult(@call_message_update_rule(node = Double, target = :out, m = (in = [1.0, 2.0],)))
2-element Vector{Float64}:
 2.0
 4.0
```

```@docs
MessagePassingRulesBase.buffer_like
```

## Scratch

A rule that needs working memory declares how to build it from its inputs, and takes it as its
`scratch`:

```julia
@define_message_update_rule(
    node = Summing, target = :out, args = (m[:in]::Vector{Float64},),
    scratch = (args) -> (work = similar(args.m[:in]),),
    body = (scratch, args) -> (scratch.work .= 2 .* args.m[:in]; sum(scratch.work)),
)
```

An engine keeps one scratch per outbound stream, builds it at the first call and passes the same
one to every later call, so the memory is allocated once. It is **write-before-read**: a rule
never relies on what an earlier call left in it, and the engine may drop or rebuild it at any
time, so the rule's result depends on its inputs alone and the rule stays pure. It never leaves
the rule, so a rule must not return it or a view into it, and it is never shared with another
rule, not even one of the same node. It is independent of `inplace`, and a rule may declare both,
taking `output` and then `scratch`.

## Annotations

A rule may record facts about its result beside it, keyed by symbol, with
[`annotate!`](@ref)`(ann, key, value)`, and read what its inputs arrived with as `ann.m[:x]` and
`ann.q[:x]`. Annotations never take part in dispatch. Where the rule's annotations go is the
caller's choice: an engine passes its own store, and a call by hand passes an
[`AnnotationStore`](@ref) to collect them, or nothing to drop them. A message's log scale is not
an annotation: a rule declares it with `logscale` (see [Log scales](@ref)).

```jldoctest rules
julia> struct Solver end

julia> @define_factor_node(node = Solver, type = Deterministic, interfaces = [:out, :in])

julia> @define_message_update_rule(
           node = Solver, target = :out, args = (m[:in]::Real,),
           body = (args, ann) -> (MessagePassingRulesBase.annotate!(ann, :iterations, 3); sqrt(args.m[:in])),
       )

julia> store = MessagePassingRulesBase.AnnotationStore();

julia> result = @call_message_update_rule(node = Solver, target = :out, m = (in = 4.0,), ann = store);

julia> MessagePassingRulesBase.getannotation(getannotations(result), :iterations)
3
```

```@docs
MessagePassingRulesBase.RuleAnnotations
MessagePassingRulesBase.AnnotationStore
MessagePassingRulesBase.NoAnnotations
MessagePassingRulesBase.annotate!
MessagePassingRulesBase.hasannotation
MessagePassingRulesBase.getannotation
```

## Working types

A rule may compute in a type chosen for the arithmetic rather than the reader, and declare how
it converts to the type users expect, which an engine applies to every marginal it forms.

```@docs
public_equivalent
```
