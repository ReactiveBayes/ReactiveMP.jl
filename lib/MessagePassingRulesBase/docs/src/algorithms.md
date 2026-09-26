```@meta
CurrentModule = MessagePassingRulesBase
```

# Algorithms and dependencies

A rule belongs to an algorithm, and the algorithm, with the graph's factorisation, decides what
each rule consumes. Almost every node runs under the default algorithm and declares nothing: an
algorithm is **not** an inference scheme. It selects which rules run and carries their
parameters.

```@docs
AbstractAlgorithm
```

## The default scheme

Under [`DefaultAlgorithm`](@ref), which minimises the Bethe free energy, whether a rule is belief
propagation, variational message passing or their structured form follows from the
factorisation, not from the rule. The factorisation splits a node's interfaces into clusters,
and an engine gives the rule for a target:

- the **messages** on the other interfaces of the target's own cluster;
- the **marginals** of the other clusters, a joint marginal for a cluster of several interfaces.

For `NormalMeanVariance` with interfaces `out`, `μ` and `v`:

| factorisation | the rule towards `out` takes | which is |
|---|---|---|
| `q(out, μ, v)` | `m[:μ]`, `m[:v]` | belief propagation |
| `q(out) q(μ) q(v)` | `q[:μ]`, `q[:v]` | mean-field variational message passing |
| `q(out, μ) q(v)` | `m[:μ]`, `q[:v]` | structured variational message passing |

A marginal rule over a cluster of several interfaces, `q(out, μ)` here, takes the messages on the
cluster's members and the marginals of the other clusters; an average energy takes one marginal
per cluster. A deterministic node's clusters are always its output and the joint over its
inputs, whatever the factorisation. A rule package defines a rule for each combination of
inputs it supports, and the types of the inputs select among rules with the same shape.

```@docs
DefaultAlgorithm
MessagePassingRulesBase.default_algorithm
```

## Extending the default

A subtype of [`DefaultAlgorithmExtension`](@ref) overrides some rules, or some dependencies, of
the default, and inherits the rest. Resolution looks for the extension's own rule first and falls
back to the default's, which then runs with `DefaultAlgorithm()` in its `algo` slot, the
algorithm it was written for.

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

A direct subtype of [`AbstractAlgorithm`](@ref) instead stands alone: only its own rules and
dependencies apply to it.

```@docs
DefaultAlgorithmExtension
```

## Algorithms with parameters

An algorithm's fields are its parameters, and a rule reads them from its `algo` slot. A rule
declared with `algorithm = T` for a parametric type `T` matches every `T{…}`; a call or a graph
gives the value.

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

A node that names a parametric algorithm as its own, `algorithm = T`, binds a rule that omits
`algorithm` to the type of its default instance, `T{Nothing}` say, not to `T`. Rules and
dependencies meant for every variant declare `algorithm = T` themselves.

## Purity

A rule is pure unless declared otherwise: it mutates neither its inputs nor any state shared
beyond one call, and draws randomness only from `ctx.rng`, which the caller owns. An algorithm
that carries state of its own is impure, and says so with a method of [`ispure`](@ref); a rule
overrides its algorithm with `pure = false` or `pure = true`.

```@docs
MessagePassingRulesBase.ispure
```

## A node's own algorithm

A node whose rules ignore the factorisation declares an algorithm of its own, and what each rule
consumes, target by target. The mixtures do: `NormalMixture` runs under `NormalMixtureVMP`,
always variational, and `Mixture` under `MixtureBP`, always over messages.

```julia
@define_factor_node(
    node = NormalMixture,
    type = Stochastic,
    interfaces = [:out, :switch, :m..., :p...],
    algorithm = NormalMixtureVMP,
    dependencies = [
        :out => (q[:switch], q[:p...], q[:m...]),
        :switch => (q[:out], q[:p...], q[:m...]),
        (:m, k) => (q[:out], q[:switch], q[:p][k]),
        (:p, k) => (q[:out], q[:switch], q[:m][k]),
    ],
)
```

The inputs of a target are subscribed to in the order they are declared, which in variational
message passing is the update schedule: here each component's precision is updated before its
mean. [`@define_dependencies`](@ref) declares the same for another algorithm of an existing node.
Every target a graph connects must be declared, `target => (default,)` for one that follows the
default scheme.

```jldoctest algorithms
julia> struct Link end

julia> struct LinkVMP <: AbstractAlgorithm end

julia> @define_factor_node(node = Link, type = Stochastic, interfaces = [:out, :in], algorithm = LinkVMP)

julia> @define_dependencies(node = Link, algorithm = LinkVMP, dependencies = [:out => (q[:in],), :in => (q[:out],)])

julia> MessagePassingRulesBase.target_dependencies(MessagePassingRulesBase.dependencies_spec(Link, LinkVMP()), MessagePassingRulesBase.Target(:out))
(MessagePassingRulesBase.Dependency(:q, :in, MessagePassingRulesBase.SingleInterface()),)
```

A declaration shows itself as a table of targets and their inputs:

```@repl algorithms
using MessagePassingRulesBase # hide
struct Link end # hide
struct LinkVMP <: AbstractAlgorithm end # hide
@define_factor_node(node = Link, type = Stochastic, interfaces = [:out, :in], algorithm = LinkVMP) # hide
@define_dependencies(node = Link, algorithm = LinkVMP, dependencies = [:out => (q[:in],), :in => (q[:out],)]) # hide
MessagePassingRulesBase.dependencies_spec(Link, LinkVMP())
```

```@docs
@define_dependencies
MessagePassingRulesBase.DependenciesSpec
MessagePassingRulesBase.dependencies_spec
MessagePassingRulesBase.free_energy_partition
```

## Extending the default scheme

A rule may need one input the factorisation does not give it, while its other inputs follow the
factorisation as usual. `default` among a target's inputs stands for the default scheme's, and
the inputs beside it are added. ContinuousTransition's rule towards `a` reads `q(a)`, the
expansion point of its transformation, under both of its factorisations:

```julia
@define_dependencies(
    node = ContinuousTransition, algorithm = CTVMP,
    dependencies = [:y => (default,), :x => (default,), :a => (default, q[:a]), :W => (default,)],
)
```

An added input is a single interface's message or marginal. The engine places it among the
default scheme's inputs in interface order, and adds nothing the default scheme already gives. A
marginal added this way is consumed and never scored: the free energy is computed over the
factorisation. [`check_rules`](@ref) checks that a rule for such a target reads the added inputs;
the rest depend on the factorisation, which a rule does not know.

## How an engine reads a declaration

An engine reads a declaration through its parts: each target's inputs, each input selecting an
interface, a whole group, a group's aligned member, every member but the target's, or members a
function picks.

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

A rule that reads the message on its own edge, as an expectation-propagation rule does, has no
message to start from in a graph with a loop through that edge. The node may declare one, and the
engine sets it on the node's inbound message at activation, where nothing was set: a model's own
initialisation wins. Probit declares one for `in`:

```julia
@define_factor_node(
    node = Probit, type = Stochastic, interfaces = [:out, :in], algorithm = ProbitEP,
    initial_messages = [:in => NormalMeanPrecision(0.0, 100.0)],
)
```

It is a default for starting, not a dependency: which inputs a rule reads stays the algorithm's.
An initial message was computed by no rule, so its log scale is undefined.

```@docs
MessagePassingRulesBase.initial_messages
```
