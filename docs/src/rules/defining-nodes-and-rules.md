# [Defining nodes and rules](@id rules-defining)

Nodes, their message update rules, marginal rules and average energies are defined with the
macros of `MessagePassingRulesBase`. The engine finds a node's rules through the base package,
so a rule defined in any loaded package is found, and none of the definitions depend on the
engine: a rule is an ordinary function of its inputs, and can be called and tested on its own.

## A worked example

A deterministic node `out = in + c` for a known shift `c`, with a rule towards each of its
unknowns:

```@example defining
using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
import MessagePassingRulesBase: annotate!

struct Shift end

@define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in, :c])

@define_message_update_rule(
    node = Shift, target = :out,
    args = (m[:in]::NormalMeanVariance, m[:c]::PointMass),
    body = (args, ann) -> begin
        annotate!(ann, :logscale, 0)
        NormalMeanVariance(mean(args.m[:in]) + mean(args.m[:c]), var(args.m[:in]))
    end,
)

@define_message_update_rule(
    node = Shift, target = :in,
    args = (m[:out]::NormalMeanVariance, m[:c]::PointMass),
    body = (args) -> NormalMeanVariance(mean(args.m[:out]) - mean(args.m[:c]), var(args.m[:out])),
)

call_message_update_rule(Shift, :out; m = (in = NormalMeanVariance(1.0, 2.0), c = PointMass(3.0)))
```

A stochastic node declares an average energy for the free energy too. Here `out ~ N(μ, 1)`,
with a variational rule towards `out`:

```@example defining
struct UnitNormal end

@define_factor_node(node = UnitNormal, type = Stochastic, interfaces = [:out, (:μ, aliases = [:mean])])

@define_message_update_rule(
    node = UnitNormal, target = :out,
    args = (q[:μ]::Any,),
    body = (args) -> NormalMeanVariance(mean(args.q[:μ]), 1.0),
)

@define_average_energy(
    node = UnitNormal,
    args = (q[:out]::Any, q[:μ]::Any),
    body = (args) -> (log(2π) + var(args.q[:out]) + var(args.q[:μ]) + abs2(mean(args.q[:out]) - mean(args.q[:μ]))) / 2,
)

call_average_energy(UnitNormal; q = (out = NormalMeanVariance(0.0, 1.0), μ = PointMass(0.0)))
```

## [Nodes](@id rules-defining-nodes)

`@define_factor_node` declares a node: its type (`Stochastic` or `Deterministic`), its
interfaces in order, the first being the output, and optionally its own algorithm, its
dependencies and how it treats known inputs. An interface may have aliases,
`(:μ, aliases = [:mean])`, and a trailing `...` declares an interface group, one member per
component, as `interfaces = [:out, :switch, :m..., :p...]`. The node itself is a type or a
function: `node = +` declares the function `+` as a node.

A node may also say what it requires of the graph, which the engine checks when it creates the
node: `matched_groups = [(:m, :p)]` for groups with as many members as each other,
`min_group_length = 2` for at least two members per group, and `factorisation = :meanfield` for
a node whose rules are variational whatever the factorisation. `NormalMixture` declares all
three, so a mixture with three means and two precisions is an error, not a switch that reads
two components.

```@docs
@define_factor_node
Stochastic
Deterministic
getnodefn
MessagePassingRulesBase.NodeSpec
MessagePassingRulesBase.InterfaceSpec
MessagePassingRulesBase.nodespec
MessagePassingRulesBase.static_inputs
MessagePassingRulesBase.matched_groups
MessagePassingRulesBase.min_group_length
MessagePassingRulesBase.required_factorisation
MessagePassingRulesBase.initial_messages
MessagePassingRulesBase.interfaces
MessagePassingRulesBase.interface_groups
MessagePassingRulesBase.alias_interface
MessagePassingRulesBase.sdtype
MessagePassingRulesBase.nodefunction
```

## [Rules](@id rules-defining-rules)

A rule names its node, its target and its inputs, and gives its body as a lambda:

- `target` is an interface (`:out`), a member of a group (`(:m, k)`, with `k` in scope in the
  arguments and the body), or, for a marginal rule, a cluster (`(:out, :μ)`).
- `args` lists the inputs with their types: messages `m[:x]`, marginals `q[:x]`, joint
  marginals `q[:y, :x]` (with a group's member written `(:p, 1)`, `q[:y, (:p, 1)]`), whole
  groups `q[:m...]`, and a group's member `q[:p][k]`. A group arrives
  as a tuple in member order, with `nothing` where the rule does not take a member.
- `body` takes, by name, any of `output` (a buffer, for an in-place rule), `scratch` (its
  working memory), `algo` (the algorithm it runs under), `ctx` (the context services), `args`
  and `ann` (the annotations).
- `ctx = (:rng,)` declares the context services the body uses. `ctx.rng` is the random number
  generator the caller owns; `ctx.product(left, right)` multiplies two distributions and returns
  the product with its own log scale; `matrix_correction(ctx, default)` is the correction a rule
  applies to a matrix it builds, its own `default` when none is set.
- `pure = false` marks a rule with side effects.

A rule that reuses another's computation calls a plain helper function both share, rather than
the other rule.

### [Rules over whatever the factorisation delivers](@id rules-defining-default-args)

A node whose rules are one computation over any factorisation, such as a tensor node,
declares `default` among a rule's `args`: the rule takes whatever inputs the default scheme
delivers, and the typed inputs named beside it.

```julia
@define_message_update_rule(
    node = DiscreteTransition, target = (:T, k), args = (default, q[:a]::DirichletCollection),
    body = (args) -> contract(args, axes_of(Target(:T, k))),
)
```

The body walks its inputs with [`MessagePassingRulesBase.rule_inputs`](@ref), `key => value`
pairs: an interface by its name, a group's member as `(:T, k)`, and a joint by its key. A rule
over the marginal of any cluster names its target with a bare name, `target = members`, bound in
the body to the cluster's key. A rule with explicit inputs over the same node and target is more
specific and wins where it applies, so a fast path can sit on top. Missing or mistyped typed
inputs make the lookup a `RuleNotFound`. There is at most one `default` rule per node, target and
algorithm; `check_rules` checks its typed inputs only.

```@docs
MessagePassingRulesBase.rule_inputs
MessagePassingRulesBase.default_inputs_match
```

### [Scratch](@id rules-defining-scratch)

A rule that needs working memory declares how to build it from its inputs, and takes it as its
`scratch`:

```julia
@define_message_update_rule(
    node = Summing, target = :out, args = (m[:in]::Vector{Float64},),
    scratch = (args) -> (work = similar(args.m[:in]),),
    body = (scratch, args) -> (scratch.work .= 2 .* args.m[:in]; sum(scratch.work)),
)
```

The engine keeps one scratch per outbound stream, builds it at the first call and passes the
same one to every later call, so the memory is allocated once. It is **write-before-read**: a
rule never relies on what an earlier call left in it, and the engine may drop or rebuild it at
any time, so the rule's result depends on its inputs alone and the rule stays pure. It never
leaves the rule, so a rule must not return it or a view into it, and it is never shared with
another rule, not even one of the same node. It is independent of `inplace`, and a rule may
declare both, taking `output` and then `scratch`. The table tests check the contract: a rule with
scratch is run again on a reused scratch filled with NaN, and must agree.

```@docs
@define_message_update_rule
@define_marginal_update_rule
@define_average_energy
MessagePassingRulesBase.Target
MessagePassingRulesBase.IndexedTarget
MessagePassingRulesBase.ClusterTarget
MessagePassingRulesBase.target_edge
MessagePassingRulesBase.target_index
MessagePassingRulesBase.cluster_members
MessagePassingRulesBase.RuleArgs
MessagePassingRulesBase.Messages
MessagePassingRulesBase.Marginals
MessagePassingRulesBase.canonical_cluster_keys
MessagePassingRulesBase.RuleContext
MessagePassingRulesBase.DEFAULT_CONTEXT_SERVICES
matrix_correction
MessagePassingRulesBase.buffer_like
```

## [Results](@id rules-defining-results)

A marginal rule whose cluster factorises returns a [`FactorizedCluster`](@ref) of its blocks,
each labelled with the members it covers; the engine hands each block to its members. A rule may
compute in an efficient working type, which [`public_equivalent`](@ref) converts to the type
users expect when the engine forms a marginal.

```@docs
FactorizedCluster
MessagePassingRulesBase.cluster_blocks
MessagePassingRulesBase.check_factorized_cluster
public_equivalent
```

## [Annotations](@id rules-defining-annotations)

A rule records facts about its result, such as its log scale, with
[`ReactiveMP.annotate!`](@ref)`(ann, key, value)`, and reads what its inputs arrived with from
`ann.m[:x]` and `ann.q[:x]`.

```@docs
MessagePassingRulesBase.RuleAnnotations
MessagePassingRulesBase.AnnotationStore
MessagePassingRulesBase.NoAnnotations
MessagePassingRulesBase.hasannotation
MessagePassingRulesBase.getannotation
```

## [Calling rules](@id rules-defining-calling)

Any rule can be called directly, without a graph, which is how rules are tested and explored.
`which_*` returns the rule a call would run.

```@example defining
which_message_update_rule(Shift, :in; m = (out = NormalMeanVariance(4.0, 2.0), c = PointMass(3.0)))
```

```@docs
call_message_update_rule
call_marginal_update_rule
call_average_energy
@call_message_update_rule
@call_marginal_update_rule
@call_average_energy
which_message_update_rule
which_marginal_update_rule
which_average_energy
@which_message_update_rule
@which_marginal_update_rule
@which_average_energy
message_passing_rule
message_passing_rule!
message_passing_marginalrule
message_passing_marginalrule!
message_passing_average_energy
MessagePassingRulesBase.execute_rule
MessagePassingRulesBase.rule_scratch
```
