```@meta
CurrentModule = MessagePassingRulesBase
```

# [Keyword reference](@id keyword-reference)

Every definition macro of the package takes keyword arguments only. This page lists each keyword
of each macro, with a minimal example and its output. The pages on
[defining nodes](@ref "Defining nodes"), [defining rules](@ref "Defining rules") and
[algorithms and dependencies](@ref "Algorithms and dependencies") explain the ideas behind them.

| macro | required | optional |
|---|---|---|
| [`@define_factor_node`](@ref) | `node`, `type`, `interfaces` | `algorithm`, `dependencies`, `initial_messages`, `static_inputs`, `matched_groups`, `min_group_length`, `factorisation` |
| [`@define_message_update_rule`](@ref) | `node`, `target`, `args`, `body` | `algorithm`, `logscale`, `reads_logscale`, `ctx`, `inplace`, `preallocate`, `scratch`, `pure` |
| [`@define_marginal_update_rule`](@ref) | `node`, `target`, `args`, `body` | `algorithm`, `ctx`, `inplace`, `preallocate`, `scratch`, `pure` |
| [`@define_average_energy`](@ref) | `node`, `args`, `body` | `algorithm`, `ctx`, `pure` |
| [`@define_dependencies`](@ref) | `node`, `algorithm`, `dependencies` | `free_energy_partition` |

A macro rejects an unknown or repeated keyword when it is expanded, and the error names the valid
ones. The examples use distributions from ExponentialFamily and
[`PointMass`](@ref glossary-point-mass) from BayesBase.

## [`@define_factor_node`](@id keyword-factor-node)

[`@define_factor_node`](@ref) declares a [factor node](@ref glossary-factor-node): what it is,
its [interfaces](@ref glossary-interface), and what it requires of a graph. The declaration is
data, a [`NodeSpec`](@ref), which [`nodespec`](@ref) returns.

```@setup nodes
using MessagePassingRulesBase, BayesBase, ExponentialFamily
```

### [node](@id keyword-node-node)

```@example nodes
struct Gauss end    # a type names a node
double(x) = 2x      # so does a function

@define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :v])
@define_factor_node(node = double, type = Deterministic, interfaces = [:out, :in])

MessagePassingRulesBase.interfaces(double)
```

`node` is what the node is: a type, such as `NormalMeanVariance`, or a function, such as `+`.
The same value names the node in every rule, every call and every graph. Required.

### [type](@id keyword-node-type)

```@example nodes
Gauss(μ, v) = NormalMeanVariance(μ, v)

f = MessagePassingRulesBase.nodefunction(Gauss)
f(out = 1.0, μ = 0.0, v = 2.0) ≈ logpdf(NormalMeanVariance(0.0, 2.0), 1.0)
```

`type` is [`Stochastic`](@ref) for a node with a density `f(out | inputs)`, or
[`Deterministic`](@ref) for a function `out = f(inputs)`. Required.

A [stochastic node](@ref glossary-stochastic-node) has [clusters](@ref glossary-cluster) that
follow the graph's [factorisation](@ref glossary-factorisation). Without groups, the macro also
defines its log-density, [`nodefunction`](@ref), which needs the node to be callable as a
distribution of its other interfaces, as `Gauss` is above. A
[deterministic node](@ref glossary-deterministic-node) always has two clusters, its output and
the joint over its inputs, whatever the factorisation.

### [interfaces](@id keyword-node-interfaces)

```@example nodes
struct Mixture end

@define_factor_node(
    node = Mixture,
    type = Stochastic,
    interfaces = [:out, (:switch, aliases = [:s]), :inputs...],
)

MessagePassingRulesBase.nodespec(Mixture)
```

`interfaces` is a vector of the node's interfaces, the output first by convention. Required. An
entry is one of:

- `:out`, a single interface. A name may contain underscores.
- `(:switch, aliases = [:s])`, an interface with other names a graph may use for it.
  [`alias_interface`](@ref)`(Mixture, :s)` returns `:switch`.
- `:inputs...`, a [group](@ref glossary-group): any number of members `(:inputs, 1)`,
  `(:inputs, 2)`, …, which a graph gives as a whole. [`interface_groups`](@ref) lists the groups.

A group may be empty unless [`min_group_length`](@ref keyword-node-min_group_length) says
otherwise. [Defining nodes](@ref "Defining nodes") describes interfaces and groups in full.

### [algorithm](@id keyword-node-algorithm)

```@example nodes
struct Link end
struct LinkVMP <: AbstractAlgorithm end

struct Smoothed <: DefaultAlgorithmExtension
    factor::Float64
end
struct Smoother end

@define_factor_node(node = Link, type = Stochastic, interfaces = [:out, :in], algorithm = LinkVMP)
@define_factor_node(node = Smoother, type = Deterministic, interfaces = [:out, :in], algorithm = Smoothed(0.5))

MessagePassingRulesBase.default_algorithm(Link), MessagePassingRulesBase.default_algorithm(Smoother)
```

`algorithm` is the [algorithm](@ref glossary-algorithm) the node's rules run under unless a call
or a graph gives another. It is a type, instantiated with no arguments, or a value. Default:
[`DefaultAlgorithm`](@ref)`()`, which almost every node keeps, since under it the factorisation
decides what each rule consumes. A node declares its own algorithm only when its rules ignore the
factorisation, and then usually declares its [dependencies](@ref keyword-node-dependencies) too.
[`default_algorithm`](@ref) returns it, and a rule that omits its own `algorithm` is defined for
its type. See [Algorithms and dependencies](@ref "Algorithms and dependencies").

### [dependencies](@id keyword-node-dependencies)

```@example nodes
struct Coupling end
struct CouplingVMP <: AbstractAlgorithm end

@define_factor_node(
    node = Coupling,
    type = Stochastic,
    interfaces = [:out, :in, :τ],
    algorithm = CouplingVMP,
    dependencies = [
        :out => (q[:τ], q[:in]),
        :in => (q[:τ], q[:out]),
        :τ => (q[:out], q[:in]),
    ],
)

MessagePassingRulesBase.dependencies_spec(Coupling, CouplingVMP())
```

`dependencies` declares what each rule consumes under the node's own algorithm, target by
target, in place of the [default scheme](@ref glossary-default-scheme). It is a vector of
`target => (inputs...)` pairs, in the vocabulary of [`@define_dependencies`](@ref)'s
[`dependencies`](@ref keyword-dependencies-dependencies). The inputs are subscribed to in the
order written, which under variational message passing is the update schedule: here the rules
towards `out` and `in` read `q(τ)` first. Default: none, and every target follows the default
scheme.

The declaration is for the node's own algorithm only, and takes no `free_energy_partition`.
Declare the dependencies of the node's other algorithms, or a partition, with
[`@define_dependencies`](@ref).

### [initial_messages](@id keyword-node-initial_messages)

```@example nodes
struct Threshold end
struct ThresholdEP <: AbstractAlgorithm end

@define_factor_node(
    node = Threshold,
    type = Stochastic,
    interfaces = [:out, :in],
    algorithm = ThresholdEP,
    initial_messages = [:in => NormalMeanPrecision(0.0, 100.0)],
)

MessagePassingRulesBase.initial_messages(Threshold)
```

`initial_messages` is a vector of `:name => message` pairs, the
[initial messages](@ref glossary-initial-message) an engine sets on the node's inbound messages
before inference, where the graph sets none. It is for a rule that reads the message on its own
edge, as an [expectation propagation](@ref glossary-expectation-propagation) rule does, which
would otherwise wait for a message that never comes. Each entry names a single interface, once;
a group has none. Default: `[]`.

An initial message is a starting value, not a dependency: which inputs a rule reads stays the
algorithm's. See [`initial_messages`](@ref).

### [static_inputs](@id keyword-node-static_inputs)

```@example nodes
struct Apply end   # out = f(ins...)

@define_factor_node(node = Apply, type = Deterministic, interfaces = [:out, :ins...], static_inputs = :fold)

MessagePassingRulesBase.static_inputs(Apply)
```

`static_inputs` says how the node treats inputs connected to constants and data. `:none`, the
default, treats them like any other input. `:fold` asks the engine to fold them into the node's
function, and to hold every update until they have a value. A rule then reaches the function of
the remaining inputs as [`getnodefn`](@ref)`(ctx.node, Target(:out))`, after declaring
`ctx = (:node,)`:

```@example nodes
@define_message_update_rule(
    node = Apply, target = :out, args = (m[:ins...]::PointMass,), ctx = (:node,), logscale = 0,
    body = (ctx, args) -> PointMass(getnodefn(ctx.node, MessagePassingRulesBase.Target(:out))(map(mean, args.m[:ins])...)),
)

# A stand-in for an engine's node. The graph connects `ins[2]` to the constant 3.0, which the
# engine folds into the function, so the rule sees one input.
struct EngineNode{F}
    f::F
end
MessagePassingRulesBase.getnodefn(node::EngineNode, ::MessagePassingRulesBase.Target{:out}) = node.f

ctx = MessagePassingRulesBase.RuleContext(node = EngineNode(x -> x * 3.0))
@call_message_update_rule(node = Apply, target = :out, m = (ins = (PointMass(2.0),),), ctx = ctx)
```

Which inputs are static is known only from the graph, so the engine does the folding. In
[ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/), such a node has one group, is
created with its function, `factornode(…; nodefn = f)`, and folds the members of the group
connected to a constant or to data. See [`static_inputs`](@ref).

### [matched_groups](@id keyword-node-matched_groups)

```@example nodes
struct NormalMix end

@define_factor_node(
    node = NormalMix,
    type = Stochastic,
    interfaces = [:out, :switch, :m..., :p...],
    matched_groups = [(:m, :p)],
)

MessagePassingRulesBase.matched_groups(NormalMix)
```

`matched_groups` is a vector of tuples of group names, each listing groups that must have as
many members as each other: here a mixture's means `m` and precisions `p` come in pairs. Each
tuple names two or more distinct groups of the node. Default: `[]`. An engine checks it when it
creates the node, so a graph with three means and two precisions is an error there, not a rule
silently reading fewer components. See [`matched_groups`](@ref).

### [min_group_length](@id keyword-node-min_group_length)

```@example nodes
struct Choice end

@define_factor_node(node = Choice, type = Stochastic, interfaces = [:out, :switch, :inputs...], min_group_length = 2)

MessagePassingRulesBase.min_group_length(Choice)
```

`min_group_length` is the fewest members every group of the node may have, a non-negative
integer: `2` for a choice between at least two inputs, `0` for a group that may be empty. A
value other than `1` needs the node to have a group. Default: `1`. An engine checks it when it creates the node. See
[`min_group_length`](@ref).

### [factorisation](@id keyword-node-factorisation)

```@example nodes
struct Switching end

@define_factor_node(node = Switching, type = Stochastic, interfaces = [:out, :switch, :inputs...], factorisation = :meanfield)

MessagePassingRulesBase.required_factorisation(Switching)
```

`factorisation` says which factorisations the node accepts. `:any`, the default, accepts every
one. `:meanfield` accepts only a graph that gives every interface a cluster of its own, the
[mean field](@ref glossary-mean-field), for a node whose rules are variational whatever the
factorisation. A deterministic node cannot declare `:meanfield`, since its clusters are fixed.
An engine checks it when it creates the node. See [`required_factorisation`](@ref).

## [`@define_message_update_rule`](@id keyword-message-rule)

[`@define_message_update_rule`](@ref) defines the rule for the [message](@ref glossary-message)
a node sends towards one of its interfaces. The examples in this section share two nodes:

```@example messages
using MessagePassingRulesBase, BayesBase, ExponentialFamily

struct Gaussian end   # f(out, μ, τ) = N(out | μ, 1/τ)
@define_factor_node(node = Gaussian, type = Stochastic, interfaces = [:out, :μ, :τ])

struct Sum end        # out = in₁ + in₂ + …
@define_factor_node(node = Sum, type = Deterministic, interfaces = [:out, :in...])
nothing # hide
```

### [node](@id keyword-message-node)

```@example messages
@define_message_update_rule(
    node = Gaussian,
    target = :out,
    args = (m[:μ]::PointMass, m[:τ]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanPrecision(mean(args.m[:μ]), mean(args.m[:τ])),
)

@call_message_update_rule(node = Gaussian, target = :out, m = (μ = PointMass(1.0), τ = PointMass(4.0)))
```

`node` is the node the rule belongs to, the same value its [`@define_factor_node`](@ref)
declaration names. A rule that omits [`algorithm`](@ref keyword-message-algorithm) reads the
node's declaration, so the node must be declared before the rule is loaded. Required.

### [target](@id keyword-message-target)

```@example messages
@define_message_update_rule(
    node = Sum,
    target = (:in, k),
    args = (m[:out]::Real, m[:in][!k]::Real),
    body = (args) -> args.m[:out] - sum(x for (i, x) in enumerate(args.m[:in]) if i != k),
)

@call_message_update_rule(node = Sum, target = (:in, 2), m = (out = 6.0, in = (1.0, nothing, 3.0)))
```

`target` is the interface the message goes to. Required. It is one of:

- `:out`, a single interface;
- `(:in, k)`, any member of the group `in`. The name `k` is bound to the member's index, an
  `Int`, in `body` and in the `logscale`, `preallocate` and `scratch` functions, without being
  listed among their parameters, and `args` may select by it.

Here `k` is `2`. The rule reads every member but its own, so the call gives `nothing` in the
target's place, as an engine does.

### [algorithm](@id keyword-message-algorithm)

```@example messages
struct Damped{T} <: DefaultAlgorithmExtension
    factor::T
end

@define_message_update_rule(
    node = Sum,
    target = :out,
    algorithm = Damped,
    args = (m[:in...]::Real,),
    body = (algo, args) -> algo.factor * sum(args.m[:in]),
)

@call_message_update_rule(node = Sum, target = :out, m = (in = (1.0, 2.0),), algorithm = Damped(0.5))
```

`algorithm` is the algorithm the rule is defined for, a type, or a value whose type is used. The
body reads the algorithm's value, and so its parameters, from its `algo` slot. A parametric type
`T`, as `Damped` here, matches every `T{…}`. Default: the type of the node's
[`default_algorithm`](@ref), almost always [`DefaultAlgorithm`](@ref). Naming one is for a
[`DefaultAlgorithmExtension`](@ref), which overrides some rules of the default, or for a node's
own algorithm. See [Algorithms and dependencies](@ref "Algorithms and dependencies").

### [args](@id keyword-message-args)

`args` is a tuple of the inputs the rule consumes, each `container[key]::T`, the container `m` for
a message or `q` for a [marginal](@ref glossary-marginal). A type left out is `Any`. The types
are what the rule dispatches on, so rules with the same shape differ by the types of their
inputs. Required. The body reads each input from its `args` slot, keyed as declared.

| entry | the input | read in the body as |
|---|---|---|
| `m[:μ]::T`, `q[:μ]::T` | the message or the marginal on `μ` | `args.m[:μ]`, `args.q[:μ]` |
| `q[:out, :μ]::T` | the joint marginal of the cluster `(out, μ)`, its members in interface order | `args.q[:out, :μ]` |
| `m[:in...]::T` | every member of the group `in`, each of type `T` | `args.m[:in]`, a tuple |
| `m[:in][k]::T` | the member with the target's index | `args.m[:in][k]` |
| `m[:in][!k]::T` | every member but the target's | `args.m[:in]`, `nothing` at `k` |
| `default` | whatever the default scheme delivers | [`rule_inputs`](@ref) |

Which inputs a rule receives is not the rule's choice: under the default algorithm, it takes the
messages of its target's cluster and the marginals of the other clusters. A rule package defines
a rule for each combination it supports.

**Messages and marginals.** Under the factorisation `q(out, μ) q(τ)`, the rule towards `out`
takes the message on `μ` and the marginal of `τ`, the
[structured](@ref glossary-structured-vmp) form:

```@example messages
@define_message_update_rule(
    node = Gaussian,
    target = :out,
    args = (m[:μ]::NormalMeanVariance, q[:τ]::Any),
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), var(args.m[:μ]) + inv(mean(args.q[:τ]))),
)

@call_message_update_rule(node = Gaussian, target = :out, m = (μ = NormalMeanVariance(1.0, 2.0),), q = (τ = GammaShapeRate(2.0, 4.0),))
```

**A joint marginal.** Under `q(out, μ) q(τ)`, the rule towards `τ` takes the joint marginal of
`out` and `μ`. [Variational message passing](@ref glossary-vmp) sends
``\exp \mathbb{E}_{q(y, x)}[\log \mathcal{N}(y \mid x, 1/\tau)]``, a gamma distribution in
``\tau`` with shape ``3/2`` and rate ``\mathbb{E}[(y - x)^2]/2``. A call gives a joint in
`clusters`, keyed by its members:

```@example messages
@define_message_update_rule(
    node = Gaussian,
    target = :τ,
    args = (q[:out, :μ]::MvNormalMeanCovariance,),
    body = (args) -> begin
        m, Σ = mean_cov(args.q[:out, :μ])
        GammaShapeRate(3 / 2, ((m[1] - m[2])^2 + Σ[1, 1] + Σ[2, 2] - 2Σ[1, 2]) / 2)
    end,
)

@call_message_update_rule(
    node = Gaussian, target = :τ,
    clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.0], [1.0 0.5; 0.5 2.0]),),
)
```

`q[(:out, :μ)]` is the same entry. A group's name in a cluster stands for all its members,
`q[(:in,)]`, and a single member is written with a literal index, `q[:out, (:in, 1)]`.

**A whole group.** A group arrives as a tuple in member order:

```@example messages
@define_message_update_rule(node = Sum, target = :out, args = (m[:in...]::Real,), body = (args) -> sum(args.m[:in]))

@call_message_update_rule(node = Sum, target = :out, m = (in = (1.0, 2.0, 3.0),))
```

**The target's own member.** For an indexed target `(:m, k)`, `m[:p][k]` is the member of `p`
with the same index. The variational rule of a normal mixture towards its `k`-th mean reads the
`k`-th precision, weighted by the probability of the `k`-th component:

```@example messages
struct Mix end
@define_factor_node(node = Mix, type = Stochastic, interfaces = [:out, :switch, :m..., :p...])

@define_message_update_rule(
    node = Mix,
    target = (:m, k),
    args = (q[:out]::Any, q[:switch]::Categorical, q[:p][k]::Any),
    body = (args) -> NormalMeanPrecision(mean(args.q[:out]), probvec(args.q[:switch])[k] * mean(args.q[:p][k])),
)

@call_message_update_rule(
    node = Mix, target = (:m, 2),
    q = (out = NormalMeanVariance(1.0, 1.0), switch = Categorical([0.3, 0.7]), p = (nothing, GammaShapeRate(2.0, 1.0))),
)
```

The tuple keeps every position and holds `nothing` where the selection leaves a member out, so
`args.q[:p][k]` is member `k` whatever was selected. `m[:in][!k]`, every member but the target's,
is the [target](@ref keyword-message-target) example's.

**Whatever the factorisation delivers.** `default` among the entries stands for the inputs the
default scheme delivers, whatever they are, beside the typed entries it requires. One rule then
serves every factorisation, and its body walks the inputs with [`rule_inputs`](@ref), as
`key => value` pairs. This rule returns the keys it received:

```@example messages
struct Tensor end
@define_factor_node(node = Tensor, type = Stochastic, interfaces = [:out, :a, :T...])

@define_message_update_rule(
    node = Tensor,
    target = :out,
    args = (default, q[:a]::PointMass),
    body = (args) -> map(first, (MessagePassingRulesBase.rule_inputs(Tensor, args.m)..., MessagePassingRulesBase.rule_inputs(Tensor, args.q)...)),
)

(
    getresult(@call_message_update_rule(node = Tensor, target = :out, m = (T = (PointMass(1.0), PointMass(2.0)),), q = (a = PointMass(0.5),))),
    getresult(@call_message_update_rule(node = Tensor, target = :out, q = (a = PointMass(0.5), T = (PointMass(1.0), PointMass(2.0))))),
)
```

The first call gives messages on `T` and the second gives marginals, and the same rule serves
both. A call without `q[:a]`, or with it of another type, finds no rule. A rule with
explicit inputs for the same node and target is more specific, and wins where it applies. A node
has at most one `default` rule per target and algorithm. See
[Defining rules](@ref "Defining rules").

### [body](@id keyword-message-body)

```@example messages
struct Relay end
@define_factor_node(node = Relay, type = Deterministic, interfaces = [:out, :in])

@define_message_update_rule(
    node = Relay,
    target = :out,
    args = (m[:in]::Any,),
    logscale = 0,
    body = (args, ann) -> begin
        hops = MessagePassingRulesBase.getannotation(ann.m[:in], :hops, 0)
        MessagePassingRulesBase.annotate!(ann, :hops, hops + 1)
        args.m[:in]
    end,
)

incoming = MessagePassingRulesBase.AnnotationStore()
MessagePassingRulesBase.annotate!(incoming, :hops, 2)
ann = MessagePassingRulesBase.RuleAnnotations(m = (in = incoming,), out = MessagePassingRulesBase.AnnotationStore())

result = @call_message_update_rule(node = Relay, target = :out, m = (in = NormalMeanVariance(0.0, 1.0),), ann = ann)
MessagePassingRulesBase.getannotation(getannotations(result), :hops)
```

`body` is the rule itself, an ordinary lambda returning the message. Required. Its parameters are
some of the slots below, named in this order, and only those it uses. A misspelled, repeated or
misordered slot is an error when the macro is expanded.

| slot | what it holds | declared with |
|---|---|---|
| `output` | the buffer to write into, first | [`inplace`](@ref keyword-message-inplace) |
| `scratch` | the rule's working memory | [`scratch`](@ref keyword-message-scratch) |
| `algo` | the algorithm value, and so its parameters | [`algorithm`](@ref keyword-message-algorithm) |
| `ctx` | the [`RuleContext`](@ref), read as `ctx.name` | [`ctx`](@ref keyword-message-ctx) |
| `args` | the inputs, read as `args.m[:μ]` | [`args`](@ref keyword-message-args) |
| `ann` | the annotations | no keyword |

`ann` reads the annotations that arrived with the inputs, keyed like them, as `ann.m[:in]`, and
writes the rule's own with [`annotate!`](@ref)`(ann, key, value)`. Here the rule counts the nodes a
message passed through. An engine passes its own annotations; a call by hand passes a
[`RuleAnnotations`](@ref). Annotations never take part in dispatch.

### [logscale](@id keyword-message-logscale)

```@example messages
struct Gain end   # out = a ⋅ in
@define_factor_node(node = Gain, type = Deterministic, interfaces = [:out, :in, :a])

@define_message_update_rule(
    node = Gain,
    target = :in,
    args = (m[:out]::NormalMeanVariance, m[:a]::PointMass),
    logscale = (args) -> -log(abs(mean(args.m[:a]))),
    body = (args) -> NormalMeanVariance(mean(args.m[:out]) / mean(args.m[:a]), var(args.m[:out]) / mean(args.m[:a])^2),
)

@call_message_update_rule(node = Gain, target = :in, m = (out = NormalMeanVariance(4.0, 1.0), a = PointMass(2.0)))
```

`logscale` declares the message's [log scale](@ref glossary-log-scale): the scalar with
`message = exp(logscale) · result`, for the normalised `result` the body returns. The message
towards `in` is ``\mathcal{N}(a x \mid m, v) = |a|^{-1}\, \mathcal{N}(x \mid m/a, v/a^2)``, so its
log scale is ``-\log|a|``, a function of the inputs. The keyword takes one of four forms:

- a number, `logscale = 0`, when the constant does not depend on the inputs;
- a function of the inputs over the slots `(algo, ctx, args)`, named in that order, as above;
- `from_body`, when the body returns [`with_logscale`](@ref)`(result, logscale)`;
- nothing: a rule that omits the keyword gives an [`UndefinedLogScale`](@ref) naming it.

```@example messages
getlogscale(@call_message_update_rule(node = Sum, target = (:in, 2), m = (out = 6.0, in = (1.0, nothing, 3.0))))
```

An undefined log scale errors only where a number is needed, in [`require_logscale`](@ref).
[Log scales](@ref "Log scales") explains each form, and when a rule must declare one.

### [reads_logscale](@id keyword-message-reads_logscale)

```@example messages
struct Identity end   # out = in
@define_factor_node(node = Identity, type = Deterministic, interfaces = [:out, :in])

@define_message_update_rule(
    node = Identity,
    target = :out,
    args = (m[:in]::Any,),
    reads_logscale = true,
    logscale = (args) -> require_logscale(args.logscale.m[:in]),
    body = (args) -> args.m[:in],
)

getlogscale(@call_message_update_rule(node = Identity, target = :out, m = (in = NormalMeanVariance(0.0, 1.0),), logscale = (in = -1.5,)))
```

`reads_logscale = true` says the rule reads the log scales its inbound messages arrived with, as
`args.logscale.m[:in]`. Here the message towards `out` is the inbound message itself, constant
included. Default: `false`. Its caller must provide them: an engine does when it tracks log
scales, and a call by hand takes them as `logscale = (in = …,)`. Without them the call is an
error:

```@example messages
try
    @call_message_update_rule(node = Identity, target = :out, m = (in = NormalMeanVariance(0.0, 1.0),))
catch err
    showerror(stdout, err)
end
```

See [Log scales](@ref "Log scales").

### [ctx](@id keyword-message-ctx)

```@example messages
using Random

struct Square end   # out = in²
@define_factor_node(node = Square, type = Deterministic, interfaces = [:out, :in])

@define_message_update_rule(
    node = Square,
    target = :out,
    args = (m[:in]::NormalMeanVariance,),
    ctx = (:rng,),
    body = (ctx, args) -> begin
        samples = abs2.(rand(ctx.rng, args.m[:in], 10_000))
        NormalMeanVariance(mean(samples), var(samples))
    end,
)

ctx = MessagePassingRulesBase.RuleContext(rng = Xoshiro(1))
getresult(@call_message_update_rule(node = Square, target = :out, m = (in = NormalMeanVariance(1.0, 0.5),), ctx = ctx))
```

`ctx` lists the context [services](@ref glossary-service) the rule reads, as a tuple of symbols,
`ctx = (:rng,)`. The body reads each from its `ctx` slot as `ctx.name`. Default: `()`, none. An
engine supplies `node`, `rng` and `matrix_correction` ([`DEFAULT_CONTEXT_SERVICES`](@ref)), and
any other name is allowed for a service of the caller's own. An engine refuses a rule whose
services its context lacks ([`check_services`](@ref)); a call by hand does not check, and a
missing service reads as `nothing`. See [The rule context](@ref "The rule context").

### [inplace](@id keyword-message-inplace)

```@example messages
struct Double end   # out = 2 ⋅ in
@define_factor_node(node = Double, type = Deterministic, interfaces = [:out, :in])

@define_message_update_rule(
    node = Double,
    target = :out,
    args = (m[:in]::Vector{Float64},),
    inplace = true,
    preallocate = (args) -> MessagePassingRulesBase.buffer_like(args.m[:in]),
    body = (output, args) -> (output .= 2 .* args.m[:in]),
)

getresult(@call_message_update_rule(node = Double, target = :out, m = (in = [1.0, 2.0],)))
```

`inplace = true` makes an [in-place rule](@ref glossary-in-place-rule): the body writes its result
into a buffer it is given, its `output` slot, first, and returns it. It needs
[`preallocate`](@ref keyword-message-preallocate). Default: `false`. An engine may keep the buffer
between calls; [`buffer_like`](@ref) builds storage of the right kind from an input.

### [preallocate](@id keyword-message-preallocate)

```@example messages
struct Every <: DefaultAlgorithmExtension   # keep every n-th entry
    n::Int
end

struct Downsample end
@define_factor_node(node = Downsample, type = Deterministic, interfaces = [:out, :in])

@define_message_update_rule(
    node = Downsample,
    target = :out,
    algorithm = Every,
    args = (m[:in]::Vector{Float64},),
    inplace = true,
    preallocate = (algo, args) -> similar(args.m[:in], cld(length(args.m[:in]), algo.n)),
    body = (output, algo, args) -> (output .= args.m[:in][1:algo.n:end]),
)

getresult(@call_message_update_rule(node = Downsample, target = :out, m = (in = [1.0, 2.0, 3.0, 4.0, 5.0],), algorithm = Every(2)))
```

`preallocate` builds an in-place rule's buffer, a function over the slots `(algo, ctx, args)`,
named in that order, and only those it uses. Here the buffer's length depends on the algorithm's
parameter as well as on the input. For an indexed target, `k` is bound in it too. Allowed only
with `inplace = true`, and then required.

### [scratch](@id keyword-message-scratch)

```@example messages
struct Norm end   # out = ‖in‖
@define_factor_node(node = Norm, type = Deterministic, interfaces = [:out, :in])

@define_message_update_rule(
    node = Norm,
    target = :out,
    args = (m[:in]::Vector{Float64},),
    scratch = (args) -> (squares = similar(args.m[:in]),),
    body = (scratch, args) -> begin
        scratch.squares .= abs2.(args.m[:in])
        sqrt(sum(scratch.squares))
    end,
)

getresult(@call_message_update_rule(node = Norm, target = :out, m = (in = [3.0, 4.0],)))
```

`scratch` builds the rule's [scratch](@ref glossary-scratch), working memory given to the body as
its `scratch` slot: a function over the slots `(algo, ctx, args)`, named in that order. The body
takes `scratch` exactly when the rule declares it. Default: none.

An engine keeps one scratch per outbound stream and reuses it, so the memory is allocated once.
It is **write-before-read**: it carries nothing between calls, and the engine may keep, drop or
rebuild it at any time. It never leaves the rule: the body does not return it or a view into
it. A rule with scratch stays pure. A rule may declare both `inplace` and `scratch`, and its body then takes `output`, then `scratch`.
A builder whose result type infers from the inputs, made of `similar` or
`zeros(eltype(...), ...)`, gives the rule a concretely typed scratch ([`rule_scratch_type`](@ref)).
See [Defining rules](@ref "Defining rules").

### [pure](@id keyword-message-pure)

```@example messages
struct Counted <: DefaultAlgorithmExtension   # counts the calls of its rules
    calls::Base.RefValue{Int}
end
MessagePassingRulesBase.ispure(::Type{Counted}) = false

@define_message_update_rule(
    node = Sum, target = :out, algorithm = Counted, args = (m[:in...]::Real,),
    body = (algo, args) -> (algo.calls[] += 1; sum(args.m[:in])),
)

@define_message_update_rule(
    node = Sum, target = (:in, k), algorithm = Counted, args = (m[:out]::Real, m[:in][!k]::Real),
    pure = true,
    body = (args) -> args.m[:out] - sum(x for x in args.m[:in] if x !== nothing),
)

counted = Counted(Ref(0))
(
    which_message_update_rule(Sum, :out; m = (in = (1.0, 2.0),), algorithm = counted).pure,
    which_message_update_rule(Sum, (:in, 1); m = (out = 3.0, in = (nothing, 2.0)), algorithm = counted).pure,
)
```

`pure` overrides the purity of the rule's algorithm: `false` for a rule with side effects, `true`
for a pure rule under an impure algorithm. Default: the algorithm's [`ispure`](@ref), `true` for
almost every algorithm. A pure rule mutates neither its inputs nor state shared beyond one call,
and draws randomness only from `ctx.rng`. `Counted` is impure, so the first rule is too; the
second touches no counter and says so. Purity is declared, not proved: an engine's purity audit
reads the flag. See [Algorithms and dependencies](@ref "Algorithms and dependencies").

## [`@define_marginal_update_rule`](@id keyword-marginal-rule)

[`@define_marginal_update_rule`](@ref) defines the rule for the joint marginal of a cluster of
several interfaces. It takes the keywords of a message rule except `logscale` and
`reads_logscale`, since a marginal carries no log scale. The examples share a normal node and a
helper that computes the joint of `out` and `μ`: for the factor
``\mathcal{N}(y \mid x + c, 1/t)`` and normal messages on ``y`` and ``x``, the joint is normal with
precision ``\begin{psmallmatrix} w_y + t & -t \\ -t & w_x + t \end{psmallmatrix}``, where ``w`` is a
message's precision.

```@example marginals
using MessagePassingRulesBase, BayesBase, ExponentialFamily

struct Gaussian end   # f(out, μ, τ) = N(out | μ, 1/τ)
@define_factor_node(node = Gaussian, type = Stochastic, interfaces = [:out, :μ, :τ])

function gaussian_joint(my, mx, t, c = 0.0)
    wy, wx = inv(var(my)), inv(var(mx))
    W = [wy+t -t; -t wx+t]
    return MvNormalMeanCovariance(W \ [wy * mean(my) + t * c, wx * mean(mx) - t * c], inv(W))
end
nothing # hide
```

### [node](@id keyword-marginal-node)

```@example marginals
@define_marginal_update_rule(
    node = Gaussian,
    target = (:out, :μ),
    args = (m[:out]::NormalMeanVariance, m[:μ]::NormalMeanVariance, q[:τ]::PointMass),
    body = (args) -> gaussian_joint(args.m[:out], args.m[:μ], mean(args.q[:τ])),
)

@call_marginal_update_rule(
    node = Gaussian, target = (:out, :μ),
    m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0)), q = (τ = PointMass(4.0),),
)
```

`node` is the node the rule belongs to, as for a [message rule](@ref keyword-message-node).
Required.

### [target](@id keyword-marginal-target)

`target` is the cluster whose joint marginal the rule computes. Required. It is one of:

- `(:out, :μ)`, a cluster of interfaces, its members in interface order, as above;
- `(:out, (:in, 1))`, a cluster with some of a group's members, written with a literal index;
- `(:out, :in)`, a cluster with every member of a group, which names the group once, whatever
  its length. A group of one member in a cluster is the whole group, so the cluster is
  `(:out, :in)`, never `(:out, (:in, 1))`; a rule written for the latter is valid, and an
  interactive call reaches it, but a graph never asks for it, so another rule runs or none does;
- a bare name, `target = members`: any cluster of the node, the name bound to the cluster's key
  in `body` and in the `preallocate` and `scratch` functions. It goes with `default` in `args`,
  for one rule over every factorisation.

A noisy sum ``\mathcal{N}(y \mid x_1 + x_2, 1/\tau)`` under the factorisation
`q(out, in₁) q(in₂) q(τ)` has the cluster `(out, in₁)`. Its rule reads the messages on the
cluster's members and the marginals of the rest, which the group `in` splits between them:

```@example marginals
struct NoisySum end   # f(out, in..., τ) = N(out | Σ in, 1/τ)
@define_factor_node(node = NoisySum, type = Stochastic, interfaces = [:out, :in..., :τ])

@define_marginal_update_rule(
    node = NoisySum,
    target = (:out, (:in, 1)),
    args = (m[:out]::NormalMeanVariance, m[:in...]::Any, q[:in...]::Any, q[:τ]::Any),
    body = (args) -> gaussian_joint(args.m[:out], args.m[:in][1], mean(args.q[:τ]), sum(mean(x) for x in args.q[:in] if x !== nothing)),
)

getresult(@call_marginal_update_rule(
    node = NoisySum, target = (:out, (:in, 1)),
    m = (out = NormalMeanVariance(3.0, 1.0), in = (NormalMeanVariance(0.0, 1.0), nothing)),
    q = (in = (nothing, NormalMeanVariance(1.0, 1.0)), τ = GammaShapeRate(2.0, 1.0)),
))
```

A rule over any cluster binds the cluster's key. This one returns the key and the inputs it
received:

```@example marginals
struct Tensor end
@define_factor_node(node = Tensor, type = Stochastic, interfaces = [:out, :a, :T...])

@define_marginal_update_rule(
    node = Tensor,
    target = members,
    args = (default, q[:a]::PointMass),
    body = (args) -> (members, map(first, MessagePassingRulesBase.rule_inputs(Tensor, args.m))),
)

(
    getresult(@call_marginal_update_rule(node = Tensor, target = (:out, (:T, 1)), m = (out = PointMass(1.0), T = (PointMass(2.0), nothing)), q = (a = PointMass(0.5),))),
    getresult(@call_marginal_update_rule(node = Tensor, target = (:out, :T), m = (out = PointMass(1.0), T = (PointMass(2.0), PointMass(3.0))), q = (a = PointMass(0.5),))),
)
```

### [algorithm](@id keyword-marginal-algorithm)

```@example marginals
struct Tempered{T} <: DefaultAlgorithmExtension   # the likelihood raised to the power β
    β::T
end

@define_marginal_update_rule(
    node = Gaussian,
    target = (:out, :μ),
    algorithm = Tempered,
    args = (m[:out]::NormalMeanVariance, m[:μ]::NormalMeanVariance, q[:τ]::PointMass),
    body = (algo, args) -> gaussian_joint(args.m[:out], args.m[:μ], algo.β * mean(args.q[:τ])),
)

getresult(@call_marginal_update_rule(
    node = Gaussian, target = (:out, :μ), algorithm = Tempered(0.5),
    m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0)), q = (τ = PointMass(4.0),),
))
```

`algorithm` is the algorithm the rule is defined for, as for a
[message rule](@ref keyword-message-algorithm): a type, or a value whose type is used, a
parametric type matching every instance. Default: the type of the node's
[`default_algorithm`](@ref).

### [args](@id keyword-marginal-args)

```@example marginals
@define_marginal_update_rule(
    node = Gaussian,
    target = (:out, :μ),
    args = (m[:out]::NormalMeanVariance, m[:μ]::NormalMeanVariance, q[:τ]::GammaShapeRate),
    body = (args) -> gaussian_joint(args.m[:out], args.m[:μ], mean(args.q[:τ])),
)

getresult(@call_marginal_update_rule(
    node = Gaussian, target = (:out, :μ),
    m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0)), q = (τ = GammaShapeRate(8.0, 2.0),),
))
```

`args` lists the inputs in the vocabulary of a [message rule's](@ref keyword-message-args). A
marginal rule typically reads the messages on the cluster's members and the marginals of the
node's other clusters. Required. Here the type of `q(τ)` selects this rule over the
[node](@ref keyword-marginal-node) example's, which takes a point mass.

### [body](@id keyword-marginal-body)

```@example marginals
@define_marginal_update_rule(
    node = Gaussian,
    target = (:out, :μ),
    args = (m[:out]::PointMass, m[:μ]::NormalMeanVariance, q[:τ]::Any),
    body = (args, ann) -> begin
        y, mx, t = mean(args.m[:out]), args.m[:μ], mean(args.q[:τ])
        w = inv(var(mx)) + t
        MessagePassingRulesBase.annotate!(ann, :observed, :out)
        FactorizedCluster((:out,) => args.m[:out], (:μ,) => NormalMeanPrecision((mean(mx) / var(mx) + t * y) / w, w))
    end,
)

store = MessagePassingRulesBase.AnnotationStore()
result = @call_marginal_update_rule(
    node = Gaussian, target = (:out, :μ),
    m = (out = PointMass(1.0), μ = NormalMeanVariance(0.0, 2.0)), q = (τ = PointMass(4.0),), ann = store,
)
getresult(result), MessagePassingRulesBase.getannotation(store, :observed)
```

`body` is the rule itself, a lambda returning the joint marginal, over the slots of a
[message rule's](@ref keyword-message-body) body: `output`, `scratch`, `algo`, `ctx`, `args` and
`ann`, named in that order. Required. With the output observed, the joint of `out` and `μ`
factorises into the point mass on `out` and the marginal of `μ`, the product of its message and
the likelihood of the observation. The rule returns a [`FactorizedCluster`](@ref) of the blocks,
and records which member was observed in its annotations.

### [ctx](@id keyword-marginal-ctx)

```@example marginals
@define_marginal_update_rule(
    node = Gaussian,
    target = (:out, :μ),
    args = (m[:out]::NormalMeanPrecision, m[:μ]::NormalMeanPrecision, q[:τ]::PointMass),
    ctx = (:jitter,),
    body = (ctx, args) -> begin
        joint = gaussian_joint(convert(NormalMeanVariance, args.m[:out]), convert(NormalMeanVariance, args.m[:μ]), mean(args.q[:τ]))
        MvNormalMeanCovariance(mean(joint), cov(joint) + ctx.jitter * [1.0 0.0; 0.0 1.0])
    end,
)

getresult(@call_marginal_update_rule(
    node = Gaussian, target = (:out, :μ),
    m = (out = NormalMeanPrecision(1.0, 1.0), μ = NormalMeanPrecision(0.0, 0.5)), q = (τ = PointMass(4.0),),
    ctx = MessagePassingRulesBase.RuleContext(jitter = 1e-6),
))
```

`ctx` lists the services the rule reads, as for a [message rule](@ref keyword-message-ctx). Here
`jitter` is a service of the caller's own, added to the covariance's diagonal. Default: `()`.

### [inplace](@id keyword-marginal-inplace)

```@example marginals
struct Transition end   # f(out, in, A) = A[out, in]
@define_factor_node(node = Transition, type = Stochastic, interfaces = [:out, :in, :A])

@define_marginal_update_rule(
    node = Transition,
    target = (:out, :in),
    args = (m[:out]::Categorical, m[:in]::Categorical, q[:A]::PointMass),
    inplace = true,
    preallocate = (args) -> similar(mean(args.q[:A])),
    body = (output, args) -> begin
        output .= probvec(args.m[:out]) .* mean(args.q[:A]) .* probvec(args.m[:in])'
        output ./= sum(output)
    end,
)

getresult(@call_marginal_update_rule(
    node = Transition, target = (:out, :in),
    m = (out = Categorical([0.5, 0.5]), in = Categorical([0.2, 0.8])), q = (A = PointMass([0.9 0.1; 0.1 0.9]),),
))
```

`inplace = true` makes the rule write the joint into a buffer, its `output` slot, first, as for a
[message rule](@ref keyword-message-inplace). The joint of two discrete interfaces is the matrix
``m_{out}(i)\, A_{ij}\, m_{in}(j)``, normalised. Default: `false`.

### [preallocate](@id keyword-marginal-preallocate)

`preallocate` builds an in-place marginal rule's buffer, a function over the slots
`(algo, ctx, args)`, as for a [message rule](@ref keyword-message-preallocate). The
[inplace](@ref keyword-marginal-inplace) example builds a matrix the size of `A`. For a rule over
any cluster, the bare target name is bound in it. This rule, for a node over binary interfaces,
builds an array with one axis per member of the cluster, and fills it with a uniform joint to show
the shape:

```@example marginals
struct Grid end
@define_factor_node(node = Grid, type = Stochastic, interfaces = [:out, :x, :y])

@define_marginal_update_rule(
    node = Grid,
    target = members,
    args = (default,),
    inplace = true,
    preallocate = (args) -> zeros(ntuple(_ -> 2, length(members))),
    body = (output, args) -> (output .= 1 / length(output)),
)

getresult(@call_marginal_update_rule(node = Grid, target = (:x, :y), m = (x = PointMass(1), y = PointMass(2)), q = (out = PointMass(0),)))
```

Allowed only with `inplace = true`, and then required.

### [scratch](@id keyword-marginal-scratch)

```@example marginals
struct Emission end   # f(out, in, A) = A[out, in]
@define_factor_node(node = Emission, type = Stochastic, interfaces = [:out, :in, :A])

@define_marginal_update_rule(
    node = Emission,
    target = (:out, :in),
    args = (m[:out]::Categorical, m[:in]::Categorical, q[:A]::PointMass),
    scratch = (args) -> (weights = similar(mean(args.q[:A])),),
    body = (scratch, args) -> begin
        scratch.weights .= probvec(args.m[:out]) .* mean(args.q[:A]) .* probvec(args.m[:in])'
        scratch.weights ./ sum(scratch.weights)
    end,
)

getresult(@call_marginal_update_rule(
    node = Emission, target = (:out, :in),
    m = (out = Categorical([0.5, 0.5]), in = Categorical([0.2, 0.8])), q = (A = PointMass([0.9 0.1; 0.1 0.9]),),
))
```

`scratch` builds the rule's working memory, as for a [message rule](@ref keyword-message-scratch).
The body returns a new matrix, never the scratch itself, which the engine reuses. Default: none.

### [pure](@id keyword-marginal-pure)

```@example marginals
const joints_computed = Ref(0)

@define_marginal_update_rule(
    node = Gaussian,
    target = (:out, :μ, :τ),
    args = (m[:out]::PointMass, m[:μ]::PointMass, m[:τ]::PointMass),
    pure = false,
    body = (args) -> (joints_computed[] += 1; FactorizedCluster((:out,) => args.m[:out], (:μ,) => args.m[:μ], (:τ,) => args.m[:τ])),
)

which_marginal_update_rule(Gaussian, (:out, :μ, :τ); m = (out = PointMass(1.0), μ = PointMass(0.0), τ = PointMass(4.0)))
```

`pure = false` declares a rule with side effects, here a global counter; `pure = true` declares a
pure rule under an impure algorithm, as for a [message rule](@ref keyword-message-pure). Default:
the algorithm's [`ispure`](@ref).

## [`@define_average_energy`](@id keyword-average-energy)

[`@define_average_energy`](@ref) defines a node's [average energy](@ref glossary-average-energy),
``\mathbb{E}_q[-\log f]`` under the marginals of its clusters: its term of the
[Bethe free energy](@ref glossary-bethe-free-energy). It has no target, returns a number and has
no log scale, so it takes neither `target`, `inplace`, `preallocate`, `scratch`, `logscale` nor
`reads_logscale`. The examples share a normal node and its energy under the mean field,

```math
U = \tfrac{1}{2}\log 2\pi - \tfrac{1}{2}\mathbb{E}[\log \tau]
  + \tfrac{1}{2}\mathbb{E}[\tau]\, \mathbb{E}\big[(y - x)^2\big].
```

```@example energies
using MessagePassingRulesBase, BayesBase, ExponentialFamily

struct Gaussian end   # f(out, μ, τ) = N(out | μ, 1/τ)
@define_factor_node(node = Gaussian, type = Stochastic, interfaces = [:out, :μ, :τ])

# E[(y - x)²] from the means and variances of y and x, and their covariance.
gaussian_energy(τ, my, vy, mx, vx, c = 0.0) = (log(2π) - mean(log, τ) + mean(τ) * ((my - mx)^2 + vy + vx - 2c)) / 2
nothing # hide
```

### [node](@id keyword-energy-node)

```@example energies
@define_average_energy(
    node = Gaussian,
    args = (q[:out]::Any, q[:μ]::Any, q[:τ]::GammaShapeRate),
    body = (args) -> gaussian_energy(args.q[:τ], mean(args.q[:out]), var(args.q[:out]), mean(args.q[:μ]), var(args.q[:μ])),
)

@call_average_energy(
    node = Gaussian,
    q = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0), τ = GammaShapeRate(2.0, 1.0)),
)
```

`node` is the node the energy belongs to, as for a [message rule](@ref keyword-message-node).
Required.

### [algorithm](@id keyword-energy-algorithm)

```@example energies
struct Tempered{T} <: DefaultAlgorithmExtension   # the likelihood raised to the power β
    β::T
end

@define_average_energy(
    node = Gaussian,
    algorithm = Tempered,
    args = (q[:out]::Any, q[:μ]::Any, q[:τ]::GammaShapeRate),
    body = (algo, args) -> algo.β * gaussian_energy(args.q[:τ], mean(args.q[:out]), var(args.q[:out]), mean(args.q[:μ]), var(args.q[:μ])),
)

getresult(@call_average_energy(
    node = Gaussian, algorithm = Tempered(0.5),
    q = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0), τ = GammaShapeRate(2.0, 1.0)),
))
```

`algorithm` is the algorithm the energy is defined for, as for a
[message rule](@ref keyword-message-algorithm). The energy of the likelihood raised to the power
``\beta`` is ``\beta`` times the energy. Default: the type of the node's
[`default_algorithm`](@ref).

### [args](@id keyword-energy-args)

```@example energies
@define_average_energy(
    node = Gaussian,
    args = (q[:out, :μ]::MvNormalMeanCovariance, q[:τ]::GammaShapeRate),
    body = (args) -> begin
        m, Σ = mean_cov(args.q[:out, :μ])
        gaussian_energy(args.q[:τ], m[1], Σ[1, 1], m[2], Σ[2, 2], Σ[1, 2])
    end,
)

getresult(@call_average_energy(
    node = Gaussian,
    clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.0], [1.0 0.5; 0.5 2.0]),),
    q = (τ = GammaShapeRate(2.0, 1.0),),
))
```

`args` lists the marginals the energy reads, one per cluster of the factorisation, as entries
`q[key]::T`. Required. An entry is one of:

- `q[:μ]::T`, the marginal of an interface that is a cluster of its own;
- `q[:out, :μ]::T`, the joint marginal of a cluster, its members in interface order, as above;
- `q[:in...]::T`, every member of a group, each a cluster of its own, as a tuple;
- `default`, whatever clusters the factorisation delivers, walked with [`rule_inputs`](@ref).

### [body](@id keyword-energy-body)

```@example energies
@define_average_energy(
    node = Gaussian,
    args = (q[:out]::Any, q[:μ]::Any, q[:τ]::PointMass),
    body = (args, ann) -> begin
        quadratic = mean(args.q[:τ]) * ((mean(args.q[:out]) - mean(args.q[:μ]))^2 + var(args.q[:out]) + var(args.q[:μ])) / 2
        MessagePassingRulesBase.annotate!(ann, :quadratic, quadratic)
        (log(2π) - log(mean(args.q[:τ]))) / 2 + quadratic
    end,
)

store = MessagePassingRulesBase.AnnotationStore()
result = @call_average_energy(
    node = Gaussian, q = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0), τ = PointMass(2.0)), ann = store,
)
getresult(result), MessagePassingRulesBase.getannotation(store, :quadratic)
```

`body` is the energy, a lambda returning a real number. Required. Its parameters are some of the
slots `algo`, `ctx`, `args` and `ann`, named in that order: an energy has no `output` and no
`scratch`. Here it records its quadratic term in its annotations, beside the result.

### [ctx](@id keyword-energy-ctx)

```@example energies
using Random

@define_average_energy(
    node = Gaussian,
    args = (q[:out]::Any, q[:μ]::Any, q[:τ]::Gamma),   # Gamma(shape, scale)
    ctx = (:rng,),
    body = (ctx, args) -> begin
        n = 10_000
        y, x, τ = rand(ctx.rng, args.q[:out], n), rand(ctx.rng, args.q[:μ], n), rand(ctx.rng, args.q[:τ], n)
        -mean(logpdf.(NormalMeanPrecision.(x, τ), y))
    end,
)

getresult(@call_average_energy(
    node = Gaussian,
    q = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0), τ = Gamma(2.0, 1.0)),
    ctx = MessagePassingRulesBase.RuleContext(rng = Xoshiro(1)),
))
```

`ctx` lists the services the energy reads, as for a [message rule](@ref keyword-message-ctx). Here
the energy is a Monte Carlo estimate, drawn with the caller's random number generator.
`Gamma(2.0, 1.0)` is the distribution of the [node](@ref keyword-energy-node) example's
`GammaShapeRate(2.0, 1.0)`, whose exact energy it approximates. Default: `()`.

### [pure](@id keyword-energy-pure)

```@example energies
const evaluations = Ref(0)

@define_average_energy(
    node = Gaussian,
    args = (q[:out]::PointMass, q[:μ]::PointMass, q[:τ]::PointMass),
    pure = false,
    body = (args) -> begin
        evaluations[] += 1
        -logpdf(NormalMeanPrecision(mean(args.q[:μ]), mean(args.q[:τ])), mean(args.q[:out]))
    end,
)

which_average_energy(Gaussian; q = (out = PointMass(1.0), μ = PointMass(0.0), τ = PointMass(4.0)))
```

`pure = false` declares an energy with side effects, here a global counter of its evaluations;
`pure = true` declares a pure energy under an impure algorithm, as for a
[message rule](@ref keyword-message-pure). Default: the algorithm's [`ispure`](@ref).

## [`@define_dependencies`](@id keyword-dependencies)

[`@define_dependencies`](@ref) declares what a node's rules consume under an algorithm, target by
target, its [dependencies](@ref glossary-dependencies), in place of the
[default scheme](@ref glossary-default-scheme). It is for an algorithm whose rules ignore the
factorisation, such as a node's own or one of its variants.
[`dependencies_spec`](@ref) returns the declaration.

```@setup dependencies
using MessagePassingRulesBase, BayesBase, ExponentialFamily
```

### [node](@id keyword-dependencies-node)

```@example dependencies
struct Link end
struct LinkVMP <: AbstractAlgorithm end
@define_factor_node(node = Link, type = Stochastic, interfaces = [:out, :in])

@define_dependencies(node = Link, algorithm = LinkVMP, dependencies = [:out => (q[:in],), :in => (q[:out],)])

MessagePassingRulesBase.dependencies_spec(Link, LinkVMP())
```

`node` is the node, which must be declared first: the declaration is checked against its
interfaces when it is loaded. Required.

### [algorithm](@id keyword-dependencies-algorithm)

```@example dependencies
struct Blend{S} <: AbstractAlgorithm   # a variant for each strategy S
    strategy::S
end

@define_dependencies(node = Link, algorithm = Blend, dependencies = [:out => (m[:in],), :in => (m[:out],)])

MessagePassingRulesBase.dependencies_spec(Link, Blend(:fast)) === MessagePassingRulesBase.dependencies_spec(Link, Blend(2))
```

`algorithm` is the algorithm the declaration is for, a type, or a value whose type is used. A
parametric type `T` covers every `T{…}`, as `Blend` does here. Required. A node's own algorithm
can also be declared inline, with the node's [`dependencies`](@ref keyword-node-dependencies)
keyword. A [`DefaultAlgorithmExtension`](@ref) that declares none uses the declaration for
[`DefaultAlgorithm`](@ref), if there is one.

### [dependencies](@id keyword-dependencies-dependencies)

`dependencies` is a vector of `target => (inputs...)` pairs, one per target. Required. A target
is `:out`, or `(:m, k)` for every member of the group `m`, which binds `k` for the inputs to
select by. The inputs are written as in a rule's [`args`](@ref keyword-message-args), without
types:

| input | what the target consumes |
|---|---|
| `m[:μ]`, `q[:μ]` | the message or the marginal of an interface |
| `q[:y, :x]` | the joint marginal of a cluster, its members in interface order |
| `m[:in...]` | every member of the group `in` |
| `m[:in][k]`, `m[:in][!k]` | the member with the target's index, or every other member |
| `m[:in][select_group_members(f; arity)]` | the members `f(k)` returns, always `arity` of them |
| `default` | the default scheme's inputs, which follow the factorisation |

A target with no inputs is `target => ()`. Every target a graph connects must be declared, and
`target => (default,)` declares one that follows the default scheme. The inputs are subscribed to
in the order written, which under variational message passing is the update schedule. The
declaration is checked when it is loaded: unknown names, a group written as a single interface or
the other way round, a target or an input given twice, and a cluster out of interface order are
errors.

**Adding to the default scheme.** A rule may need an input the factorisation does not give it,
while its other inputs follow the factorisation. `:a => (default, q[:a])` is the default scheme's
inputs plus the marginal of `a`:

```@example dependencies
struct Transform end   # y = a(x), expanded around q(a)
struct TransformVMP <: AbstractAlgorithm end
@define_factor_node(node = Transform, type = Stochastic, interfaces = [:y, :x, :a], algorithm = TransformVMP)

@define_dependencies(
    node = Transform,
    algorithm = TransformVMP,
    dependencies = [:y => (default,), :x => (default,), :a => (default, q[:a])],
)

spec = MessagePassingRulesBase.dependencies_spec(Transform, TransformVMP())
```

An input added beside `default` is a single interface's message or marginal. The engine places it
among the default scheme's inputs in interface order, and a marginal added this way is consumed
without being scored. [`extends_default_scheme`](@ref) and [`target_dependencies`](@ref) read it:

```@example dependencies
target = MessagePassingRulesBase.Target(:a)
MessagePassingRulesBase.extends_default_scheme(spec, target), MessagePassingRulesBase.target_dependencies(spec, target)
```

**Selecting group members.** [`select_group_members`](@ref)`(f; arity)` selects the members
`f(k)` returns for target index `k`, always `arity` of them, so an engine knows how many inputs to
wait for before it calls `f`. Here every member reads the first one, the others' own messages
and the output:

```@example dependencies
struct Anchored end
struct AnchoredBP <: AbstractAlgorithm end
@define_factor_node(node = Anchored, type = Stochastic, interfaces = [:out, :x...], algorithm = AnchoredBP)

@define_dependencies(
    node = Anchored,
    algorithm = AnchoredBP,
    dependencies = [
        :out => (m[:x...],),
        (:x, k) => (m[:out], m[:x][select_group_members(k -> (1,); arity = 1)]),
    ],
)

spec = MessagePassingRulesBase.dependencies_spec(Anchored, AnchoredBP())
selector = last(MessagePassingRulesBase.target_dependencies(spec, MessagePassingRulesBase.IndexedTarget(:x, 3))).selector
MessagePassingRulesBase.selected_indices(selector, 3, 4), MessagePassingRulesBase.selection_arity(selector, 4)
```

An engine reads the selection through [`selected_indices`](@ref) and [`selection_arity`](@ref):
for target `(:x, 3)` of a group of four, this selector picks the one member `(1,)`. See
[Algorithms and dependencies](@ref "Algorithms and dependencies").

### [free_energy_partition](@id keyword-dependencies-free_energy_partition)

```@example dependencies
struct Nonlinear end   # out = f(in...)
struct JointInputs <: AbstractAlgorithm end
@define_factor_node(node = Nonlinear, type = Deterministic, interfaces = [:out, :in...])

@define_dependencies(
    node = Nonlinear,
    algorithm = JointInputs,
    dependencies = [:out => (m[:in...],), (:in, k) => (q[(:in,)], m[:in][k])],
    free_energy_partition = [(:out,), (:in,)],
)

MessagePassingRulesBase.free_energy_partition(MessagePassingRulesBase.dependencies_spec(Nonlinear, JointInputs()))
```

`free_energy_partition` is the partition the free energy is computed over, a vector of tuples of
interface names that covers every interface once. A group's name stands for all its members, so
here the free energy scores the output and the joint over the inputs. What a rule consumes need
not be a block of the partition: a marginal consumed outside it is never scored. An engine refuses
a graph whose factorisation is not this partition, block for block. Default: none, and the
partition is the graph's factorisation. See [`free_energy_partition`](@ref).
