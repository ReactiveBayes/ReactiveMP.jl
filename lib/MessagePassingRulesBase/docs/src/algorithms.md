```@meta
CurrentModule = MessagePassingRulesBase
```

# Algorithms and dependencies

Every [rule](@ref glossary-rule) belongs to an [algorithm](@ref glossary-algorithm). The
algorithm and the graph's [factorisation](@ref glossary-factorisation) together decide what
each rule consumes. An algorithm is **not** an inference scheme: it selects which rules run and
carries their parameters. Almost every node runs under the default algorithm and declares
nothing.

```@docs
AbstractAlgorithm
```

## The default scheme

Under [`DefaultAlgorithm`](@ref), the factorisation alone decides whether a rule is
[belief propagation](@ref glossary-belief-propagation),
[variational message passing](@ref glossary-vmp) or
[structured variational message passing](@ref glossary-structured-vmp). The rule itself does
not decide. `DefaultAlgorithm` minimises the [Bethe free energy](@ref glossary-bethe-free-energy).

The factorisation splits a node's interfaces into [clusters](@ref glossary-cluster). An engine
gives the rule for a target two kinds of input:

- the [messages](@ref glossary-message) on the other interfaces of the target's own cluster;
- the [marginals](@ref glossary-marginal) of the other clusters, a joint marginal for a cluster
  of several interfaces.

This is the [default scheme](@ref glossary-default-scheme). For `NormalMeanVariance`, with the
interfaces `out`, `μ` and `v`, it gives:

| factorisation | the rule towards `out` takes | which is |
|---|---|---|
| `q(out, μ, v)` | `m[:μ]`, `m[:v]` | belief propagation |
| `q(out) q(μ) q(v)` | `q[:μ]`, `q[:v]` | [mean-field](@ref glossary-mean-field) variational message passing |
| `q(out, μ) q(v)` | `m[:μ]`, `q[:v]` | structured variational message passing |

A marginal rule over a cluster of several interfaces, `q(out, μ)` here, takes the messages on
the cluster's members and the marginals of the other clusters. An
[average energy](@ref glossary-average-energy) takes one marginal per cluster. A
[deterministic node](@ref glossary-deterministic-node)'s clusters are always its output and the
joint over its inputs, whatever the factorisation.

A rule package defines a rule for each combination of inputs it supports. Among rules with the
same inputs, the types of the inputs select one. [Your first node](@ref tutorial-first-node)
writes the rules of one node for each row of the table.

```@docs
DefaultAlgorithm
MessagePassingRulesBase.default_algorithm
```

## Extending the default

A subtype of [`DefaultAlgorithmExtension`](@ref) overrides some rules, or some dependencies, of
the default, and inherits the rest. Resolution looks for the extension's own rule first. When
there is none, it falls back to the default's rule. That rule then runs with
`DefaultAlgorithm()` in its `algo` slot, the algorithm it was written for.

```jldoctest algorithms
julia> using MessagePassingRulesBase

julia> struct Shift end

julia> @define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in])

julia> @define_message_update_rule(node = Shift, target = :out, args = (m[:in]::Real,), body = (args) -> args.m[:in] + 1)

julia> @define_message_update_rule(node = Shift, target = :in, args = (m[:out]::Real,), body = (args) -> args.m[:out] - 1)

julia> struct Doubled <: DefaultAlgorithmExtension end

julia> @define_message_update_rule(
           node = Shift, target = :out, algorithm = Doubled, args = (m[:in]::Real,),
           body = (args) -> 2 * (args.m[:in] + 1),
       )

julia> getresult(@call_message_update_rule(node = Shift, target = :out, m = (in = 1.0,), algorithm = Doubled()))
4.0

julia> MessagePassingRulesBase.getalgorithm(@call_message_update_rule(node = Shift, target = :in, m = (out = 1.0,), algorithm = Doubled()))
DefaultAlgorithm()
```

`Doubled` has its own rule towards `out`, which the first call runs. It has no rule towards
`in`, so the second call runs the default's rule, under `DefaultAlgorithm()`.

A direct subtype of [`AbstractAlgorithm`](@ref) stands alone instead: only its own rules and
dependencies apply to it.

```@docs
DefaultAlgorithmExtension
```

## Algorithms with parameters

An algorithm's fields are its parameters, and a rule reads them from its `algo` slot. A rule
declared with `algorithm = T`, for a parametric type `T`, matches every `T{…}`. A call or a
graph gives the value.

```jldoctest algorithms
julia> struct Damped{T} <: DefaultAlgorithmExtension
           factor::T
       end

julia> @define_message_update_rule(
           node = Shift, target = :out, algorithm = Damped, args = (m[:in]::Real,),
           body = (algo, args) -> algo.factor * (args.m[:in] + 1),
       )

julia> getresult(@call_message_update_rule(node = Shift, target = :out, m = (in = 1.0,), algorithm = Damped(0.5)))
1.0
```

A node may name a parametric algorithm as its own, `algorithm = T`. A rule that omits
`algorithm` is then bound to the type of the node's default instance, `T{Nothing}` say, not to
`T`. Rules and dependencies meant for every variant declare `algorithm = T` themselves.
[A node with its own algorithm](@ref tutorial-algorithm) builds such a node step by step.

## Purity

A rule is pure unless it is declared otherwise. A pure rule mutates neither its inputs nor any
state shared beyond one call. It draws randomness only from `ctx.rng`, which the caller owns.

An algorithm that carries state of its own is impure, and it says so with a method of
[`ispure`](@ref). A rule overrides its algorithm's purity with `pure = false` or `pure = true`.

```@docs
MessagePassingRulesBase.ispure
```

## A node's own algorithm

A node whose rules ignore the factorisation declares an algorithm of its own. It also declares
the [dependencies](@ref glossary-dependencies) of each target: what the rule for that target
consumes. The mixtures do this. `NormalMixture` runs under `NormalMixtureVMP` and is always
variational. `Mixture` runs under `MixtureBP` and always reads messages.

The node below has the interfaces of `NormalMixture`, with the groups `m` and `p` for the
components' means and precisions. Every rule takes marginals:

```@example algorithms-own
using MessagePassingRulesBase

struct Blend end
struct BlendVMP <: AbstractAlgorithm end

@define_factor_node(
    node = Blend,
    type = Stochastic,
    interfaces = [:out, :switch, :m..., :p...],
    algorithm = BlendVMP,
    matched_groups = [(:m, :p)],
    dependencies = [
        :out => (q[:switch], q[:p...], q[:m...]),
        :switch => (q[:out], q[:p...], q[:m...]),
        (:m, k) => (q[:out], q[:switch], q[:p][k]),
        (:p, k) => (q[:out], q[:switch], q[:m][k]),
    ],
)

MessagePassingRulesBase.dependencies_spec(Blend, BlendVMP())
```

The declaration draws itself as a table of targets and their inputs. An engine subscribes to a
target's inputs in the order they are declared, and in variational message passing that order is
the update schedule. Here each component's precision is updated before its mean.

Every target a graph connects must be declared. A target that follows the default scheme is
declared as `target => (default,)`.

[`@define_dependencies`](@ref) declares the same for another algorithm of an existing node:

```@example algorithms-own
struct Link end
struct LinkVMP <: AbstractAlgorithm end

@define_factor_node(node = Link, type = Stochastic, interfaces = [:out, :in])

@define_dependencies(node = Link, algorithm = LinkVMP, dependencies = [:out => (q[:in],), :in => (q[:out],)])

MessagePassingRulesBase.dependencies_spec(Link, LinkVMP())
```

`Link` runs under `DefaultAlgorithm` unless a call or a graph selects `LinkVMP`. Under
`LinkVMP`, each rule reads the marginal of the other interface.

```@docs
@define_dependencies
MessagePassingRulesBase.DependenciesSpec
MessagePassingRulesBase.dependencies_spec
MessagePassingRulesBase.free_energy_partition
```

## Extending the default scheme

A rule may need one input that the factorisation does not give it, while its other inputs
follow the factorisation as usual. `default` among a target's inputs stands for the inputs of the
default scheme, and the inputs beside it are added to them. ContinuousTransition's rule towards
`a`, for example, reads `q(a)`, the expansion point of its transformation, under both of its
factorisations. The node below declares the same:

```@example algorithms-extend
using MessagePassingRulesBase

struct Transition end   # y ~ f(x; a)
struct TransitionVMP <: AbstractAlgorithm end

@define_factor_node(node = Transition, type = Stochastic, interfaces = [:y, :x, :a])

@define_dependencies(
    node = Transition, algorithm = TransitionVMP,
    dependencies = [:y => (default,), :x => (default,), :a => (default, q[:a])],
)

MessagePassingRulesBase.dependencies_spec(Transition, TransitionVMP())
```

An added input is a single interface's message or marginal. The engine places it among the
inputs of the default scheme in interface order. It adds nothing the default scheme already
gives.

A marginal added this way is consumed and never scored: the free energy is computed over the
factorisation. [`check_rules`](@ref) checks that a rule for such a target reads the added inputs.
It does not check the other inputs, which depend on the factorisation, and a rule does not know
the factorisation.

## How an engine reads a declaration

An engine reads a declaration through its parts. Each target has its inputs, a
[`TargetDependencies`](@ref). Each input selects one of these:

- an interface;
- a whole group;
- a group's member aligned with the target;
- every member but the target's;
- the members a function picks.

```@example algorithms-own
spec = MessagePassingRulesBase.dependencies_spec(Link, LinkVMP())
MessagePassingRulesBase.target_dependencies(spec, MessagePassingRulesBase.Target(:out))
```

```@docs
MessagePassingRulesBase.TargetDependencies
MessagePassingRulesBase.target_dependencies
MessagePassingRulesBase.extends_default_scheme
MessagePassingRulesBase.Dependency
MessagePassingRulesBase.DependencySelector
MessagePassingRulesBase.SingleInterface
MessagePassingRulesBase.AllGroupMembers
MessagePassingRulesBase.AlignedGroupMember
MessagePassingRulesBase.AllGroupMembersButSelf
MessagePassingRulesBase.CustomGroupSelector
MessagePassingRulesBase.select_group_members
MessagePassingRulesBase.selected_indices
MessagePassingRulesBase.selection_arity
```

## Initial messages

Some rules read the message on their own edge, as an
[expectation propagation](@ref glossary-expectation-propagation) rule does. In a graph with a
loop through that edge, such a rule has no message to start from. The node may therefore declare
an [initial message](@ref glossary-initial-message) for the edge. Probit declares one for `in`,
and the node below does the same:

```@example algorithms-initial
using MessagePassingRulesBase, ExponentialFamily

struct Threshold end   # out = (in > 0)
struct ThresholdEP <: AbstractAlgorithm end

@define_factor_node(
    node = Threshold, type = Stochastic, interfaces = [:out, :in], algorithm = ThresholdEP,
    dependencies = [:out => (m[:in],), :in => (m[:out], m[:in])],
    initial_messages = [:in => NormalMeanPrecision(0.0, 100.0)],
)

MessagePassingRulesBase.nodespec(Threshold)
```

The rule towards `in` reads `m[:in]`, the message on its own edge. The engine sets the initial
message on the node's inbound message at activation, where nothing was set, so a model's own
initialisation wins.

An initial message is a default for starting, not a dependency: the algorithm still decides
which inputs a rule reads. No rule computed an initial message, so its log scale is undefined.

```@docs
MessagePassingRulesBase.initial_messages
```
