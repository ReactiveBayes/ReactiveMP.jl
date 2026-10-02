```@meta
CurrentModule = MessagePassingRulesBase
```

# Defining rules

A node computes three kinds of thing, and you define each with its own macro:

- a [message](@ref glossary-message) towards one of its [interfaces](@ref glossary-interface),
  with [`@define_message_update_rule`](@ref);
- the joint [marginal](@ref glossary-marginal) of a [cluster](@ref glossary-cluster) of its
  interfaces, with [`@define_marginal_update_rule`](@ref);
- its [average energy](@ref glossary-average-energy), its term of the
  [Bethe free energy](@ref glossary-bethe-free-energy), with [`@define_average_energy`](@ref).

Each definition names its node and its target. An average energy has no target. The definition
also lists the inputs the rule takes, with their types, and gives its body, an ordinary lambda.

The macro turns the definition into a method of [`find_message_rule`](@ref),
[`find_marginal_rule`](@ref) or [`find_average_energy`](@ref). Julia's dispatch then finds the
rule by the node, the target, the [algorithm](@ref glossary-algorithm) and the types of the
inputs, in any loaded package. A rule that omits `algorithm` belongs to its node's default
algorithm, so you declare the node before the rule is loaded.

[Your first node](@ref tutorial-first-node) writes the rules of one node step by step. The
[Keyword reference](@ref keyword-reference) lists every keyword of the three macros.

```@docs
@define_message_update_rule
@define_marginal_update_rule
@define_average_energy
```

## Targets

A message rule's target is a single interface, `target = :out`, or any member of a
[group](@ref glossary-group), `target = (:in, k)`. The second form binds `k` to the member's
index in the body, and the inputs can select members by it.

A marginal rule's target is a cluster, `target = (:out, :μ)`, with its members in interface
order. A member of a group is written `(:T, 1)` inside a cluster.

Internally a target is a type, so rules dispatch on it:

```@docs
MessagePassingRulesBase.Target
MessagePassingRulesBase.IndexedTarget
MessagePassingRulesBase.ClusterTarget
MessagePassingRulesBase.target_edge
MessagePassingRulesBase.target_index
MessagePassingRulesBase.cluster_members
```

## Inputs

`args` lists what a rule consumes:

- `m[:x]`, the message on `x`;
- `q[:x]`, the marginal of `x`;
- `q[:y, :x]`, the joint marginal of the cluster `(y, x)`;
- `m[:in...]`, every member of the group `in`;
- `m[:in][k]`, for a group target, the member aligned with the target;
- `m[:in][!k]`, for a group target, every other member.

A group arrives as a tuple in member order. The tuple holds `nothing` where the selection leaves
a member out, so `args.m[:in][k]` is member `k` whatever was selected:

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

The body receives the inputs as a [`RuleArgs`](@ref): `args.m` is a [`Messages`](@ref), and
`args.q` is a [`Marginals`](@ref).

The rule does not choose its inputs. The engine chooses them, and under the default algorithm
they follow from the [factorisation](@ref glossary-factorisation), as
[Algorithms and dependencies](@ref) explains. You define a rule for the inputs it will be given.
[A deterministic node with a group](@ref tutorial-groups) writes group rules step by step.

```@docs
MessagePassingRulesBase.RuleArgs
MessagePassingRulesBase.Messages
MessagePassingRulesBase.Marginals
MessagePassingRulesBase.canonical_cluster_keys
```

### Rules over whatever the factorisation delivers

A tensor node, among others, computes the same thing under any factorisation. Such a node's
rule declares `default` among its `args`. The rule then takes whatever inputs the
[default scheme](@ref glossary-default-scheme) delivers, and it requires the typed inputs named
beside `default`.

The node below averages the means of its inputs `x`, whether they arrive as messages or as
marginals. Its rule requires the marginal of `v` to be a point mass:

```@example rules-default
using MessagePassingRulesBase, BayesBase, ExponentialFamily
using MessagePassingRulesBase: rule_inputs

struct Average end   # out ~ N(mean of x₁, x₂, …, v)

@define_factor_node(node = Average, type = Stochastic, interfaces = [:out, :v, :x...])

@define_message_update_rule(
    node = Average, target = :out, args = (default, q[:v]::PointMass),
    body = (args) -> begin
        inputs = (rule_inputs(Average, args.m)..., rule_inputs(Average, args.q)...)
        means = [mean(value) for (key, value) in inputs if key isa Tuple]
        NormalMeanVariance(sum(means) / length(means), mean(args.q[:v]))
    end,
)

@call_message_update_rule(
    node = Average, target = :out,
    m = (x = (NormalMeanVariance(1.0, 1.0), NormalMeanVariance(3.0, 1.0)),),
    q = (v = PointMass(2.0),),
)
```

The same rule runs when the `x` arrive as marginals:

```@example rules-default
@call_message_update_rule(
    node = Average, target = :out,
    q = (x = (NormalMeanVariance(1.0, 1.0), NormalMeanVariance(3.0, 1.0)), v = PointMass(2.0)),
)
```

The body walks its inputs with [`rule_inputs`](@ref), which returns `key => value` pairs. The
key is an interface's name, `(:x, k)` for a member of a group, or a joint's key.

A marginal rule over any cluster names its target with a bare name, `target = members`. The body
receives the cluster's key under that name.

A rule with explicit inputs for the same node and target is more specific, and it wins where it
applies. You can therefore place a fast path on top of a `default` rule. When a typed input is
missing or has another type, the lookup returns a [`RuleNotFound`](@ref). A node has at most one
`default` rule per target and algorithm.

```@docs
MessagePassingRulesBase.rule_inputs
```

## The body

The body is a lambda over some of the slots `(output, scratch, algo, ctx, args, ann)`. You name
only the slots the body uses, in that order:

- `output`: the buffer an [in-place rule](@ref glossary-in-place-rule) writes into;
- `scratch`: the rule's working memory;
- `algo`: the algorithm value the rule runs under, which carries its parameters (see
  [Algorithms and dependencies](@ref));
- `ctx`: the [`RuleContext`](@ref), which holds the [services](@ref glossary-service) the rule
  declares with `ctx = (...)` (see [The rule context](@ref));
- `args`: the inputs;
- `ann`: the annotations.

A rule may compute its result with another rule, of its own node or of another, a packaged one
included. It calls that rule with the inputs it builds and forwards its `ctx`, so the other rule
sees the same services. The node below shifts the message of the `Average` node above:

```@example rules-default
struct ShiftedAverage end   # out ~ N(1 + mean of x₁, x₂, …, v)

@define_factor_node(node = ShiftedAverage, type = Stochastic, interfaces = [:out, :v, :x...])

@define_message_update_rule(
    node = ShiftedAverage, target = :out, args = (m[:x...]::NormalMeanVariance, q[:v]::PointMass),
    body = (ctx, args) -> begin
        average = getresult(call_message_update_rule(Average, :out; m = (x = args.m[:x],), q = (v = args.q[:v],), ctx))
        NormalMeanVariance(mean(average) + 1.0, var(average))
    end,
)

@call_message_update_rule(
    node = ShiftedAverage, target = :out,
    m = (x = (NormalMeanVariance(1.0, 1.0), NormalMeanVariance(3.0, 1.0)),),
    q = (v = PointMass(2.0),),
)
```

The keyword call allocates its arguments. Where that matters, call the positional
[`message_passing_rule`](@ref)`(Average, Target(:out), algorithm, RuleArgs(m = …, q = …), ctx)`
instead, which allocates nothing, or `message_passing_marginalrule` and
`message_passing_average_energy` for the other kinds (see
[Resolving without the interactive layer](@ref)).

## In-place rules

A rule that writes its result into a buffer declares `inplace = true`. It also declares
`preallocate`, a function over the slots `(algo, ctx, args)` that builds the buffer. Its body
takes `output` first and returns it. An engine may keep the buffer between calls.
[`buffer_like`](@ref) builds storage of the right kind from an input.

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

A rule that needs working memory, its [scratch](@ref glossary-scratch), declares how to build it
from its inputs. The body then takes it as its `scratch` slot:

```@example rules-scratch
using MessagePassingRulesBase

struct Summing end   # out = 2 · sum(in), for a vector in

@define_factor_node(node = Summing, type = Deterministic, interfaces = [:out, :in])

@define_message_update_rule(
    node = Summing, target = :out, args = (m[:in]::Vector{Float64},),
    scratch = (args) -> (work = similar(args.m[:in]),),
    body = (scratch, args) -> (scratch.work .= 2 .* args.m[:in]; sum(scratch.work)),
)

@call_message_update_rule(node = Summing, target = :out, m = (in = [1.0, 2.0],))
```

An engine keeps one scratch per outbound stream. It builds the scratch at the first call and
passes the same one to every later call, so the memory is allocated once.

A scratch is **write-before-read**:

- A rule never relies on what an earlier call left in the scratch. The engine may drop or
  rebuild it at any time. The rule's result therefore depends on its inputs alone, and the rule
  stays pure.
- The scratch never leaves the rule. A rule does not return it, or a view into it.
- The scratch is never shared with another rule, not even one of the same node.

Scratch is independent of `inplace`. A rule may declare both, and its body then takes `output`
and then `scratch`.

The scratch's type depends on the types of the inputs, their element types included. An engine
that keeps the scratch between calls infers its type from them, with
[`rule_scratch_type`](@ref), and asserts the kept scratch to that type. Inputs of other types get
a scratch of their own. You declare nothing for this.

A builder whose result type infers runs the rule on a concretely typed scratch. Builders made of
`similar` or `zeros(eltype(...), ...)` over the inputs infer. A builder that does not infer, say
because it reads a global, runs the rule on an untyped scratch, which costs a dynamic call per
call.

## Annotations

A rule may record facts about its result beside it, keyed by symbol, with
[`annotate!`](@ref)`(ann, key, value)`. It reads the annotations its inputs arrived with as
`ann.m[:x]` and `ann.q[:x]`. Annotations never take part in dispatch.

The caller decides where the rule's annotations go. An engine passes its own store. A call by
hand passes an [`AnnotationStore`](@ref) to collect them, or nothing to drop them. A message's
[log scale](@ref glossary-log-scale) is not an annotation: a rule declares it with `logscale`
(see [Log scales](@ref)).

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

## [Checking inputs](@id rules-checking-inputs)

A rule's `args` say which types its inputs have. Some rules need more: a probability between 0
and 1, a vector of a given length, a univariate marginal where the rule takes `Any`. The
`args_check` keyword checks that as the body starts. It is a function of the inputs, over the
same slots as [`logscale`](@ref keyword-message-logscale), `(algo, ctx, args)`, and returns:

- `true` when the inputs pass;
- `false` when they do not, and the rule raises a [`RuleInputError`](@ref) that quotes the check's
  source;
- a string when they do not, and the error says that instead. Return one when the check combines
  several conditions, whose source would read poorly, or when the message should name the
  offending value.

```@example rules-checks
using MessagePassingRulesBase, BayesBase, ExponentialFamily

struct Coin end   # f(out, p) = Bernoulli(out | p)
@define_factor_node(node = Coin, type = Stochastic, interfaces = [:out, :p])

@define_message_update_rule(
    node = Coin, target = :out, args = (m[:p]::PointMass,),
    args_check = (args) -> 0 <= mean(args.m[:p]) <= 1 || lazy"`p` is a probability, between 0 and 1; got $(mean(args.m[:p]))",
    body = (args) -> Bernoulli(mean(args.m[:p])),
)

try
    @call_message_update_rule(node = Coin, target = :out, m = (p = PointMass(1.5),))
catch err
    print(first(split(sprint(showerror, err), "\n  rule at")))
end
```

The check runs wherever the rule runs: in an engine, in a call by hand and in a test. A failed
check is an error, not a reason to select another rule, since resolution has already chosen this
one. A combination of inputs a node does not support at all is a rule of its own whose body raises
the error, found by dispatch like any other.

**Cost.** A check that reads only the inputs' types, such as
`args_check = (args) -> variate_form(typeof(args.q[:y])) === Univariate`, costs nothing: the rule
is compiled for those types, and the check folds away. A check that reads a value costs one
comparison. The error path is outlined, so the rule allocates nothing until a check fails. A
rule with a [`preallocate`](@ref keyword-message-preallocate) or
[`scratch`](@ref keyword-message-scratch) helper runs it first: the check guards the body, not
the helpers.

**Performance trap.** Build the message only where the check fails. Placed after `||`, as above,
it is: `cond || "...$(x)"` builds nothing on a passing call, and costs what the plain condition
does. A message bound before the condition, `msg = "...$(x)"; cond || msg`, or formatted by a
helper before it decides, is built on every call and allocates there. Writing it as a
`lazy"..."` string is the safe habit: the formatting waits until the error is shown, wherever the
string is made.

## Working types

A rule may compute in a type chosen for the arithmetic rather than for the reader. It then
declares how that type converts to the type users expect, with [`public_equivalent`](@ref). An
engine applies the conversion to every marginal it forms.

```@docs
public_equivalent
```
