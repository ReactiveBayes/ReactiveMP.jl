```@meta
CurrentModule = MessagePassingRulesBase
```

# [A deterministic node with a group](@id tutorial-groups)

This tutorial builds a sum of any number of inputs,

```math
y = x_1 + x_2 + \dots + x_n,
```

where `y` is the output and the ``x_k`` are the inputs. The node is a
[deterministic node](@ref glossary-deterministic-node): its output is a function of its inputs,
with no noise. Its inputs form a [group](@ref glossary-group), one interface with any number of
members, so one declaration serves a sum of two terms and a sum of twenty.

You write its [belief propagation](@ref glossary-belief-propagation) rules for normal messages
and the joint [marginal](@ref glossary-marginal) of its inputs. The tutorial assumes you have
read [Your first node](@ref tutorial-first-node).

```@example groups
using MessagePassingRulesBase, BayesBase, ExponentialFamily, LinearAlgebra
nothing # hide
```

## Declare the node

A sum could be the function `+` itself, as StandardMessagePassingRules declares it, so that an
RxInfer model writes `y ~ x + z`. Here it is a type of its own, `Sum`, since a sum has no
distribution to name it ([What a node is](@ref tutorial-first-node) explains the choice).
[`@define_factor_node`](@ref) declares it. In its [`interfaces`](@ref keyword-node-interfaces),
a trailing `...` after a name declares a group. Its members are `(:in, 1)`, `(:in, 2)`, and so
on, and a graph gives them all at once.

```@example groups
struct Sum end

@define_factor_node(
    node = Sum,
    type = Deterministic,
    interfaces = [:out, :in...],
    min_group_length = 2,
)

MessagePassingRulesBase.nodespec(Sum)
```

[`Deterministic`](@ref) says the node computes `out` from its inputs.
[`min_group_length = 2`](@ref keyword-node-min_group_length) says a sum has at least two terms. An engine checks it when it builds the node in a graph, so a
sum with one input is an error there. Without the keyword, a group needs one member; with
`min_group_length = 0`, it may be empty. [`min_group_length`](@ref) reads it back:

```@example groups
MessagePassingRulesBase.min_group_length(Sum)
```

## A message towards the output

Each input carries a normal message, ``\mathcal{N}(x_k \mid m_k, v_k)``. The message towards
`out` is the [pushforward](@ref glossary-pushforward) of these messages through the sum, the
distribution of ``y`` when each ``x_k`` has its message as its distribution:

```math
\mu_{f \to y}(y) = \int \delta\Big(y - \sum_k x_k\Big) \prod_k \mathcal{N}(x_k \mid m_k, v_k)\, \mathrm{d}x
                 = \mathcal{N}\Big(y \;\Big|\; \sum_k m_k,\ \sum_k v_k\Big).
```

The means add and the variances add. [`@define_message_update_rule`](@ref) defines the rule, and
its [`args`](@ref keyword-message-args) read every member of the group with `m[:in...]`:

```@example groups
@define_message_update_rule(
    node = Sum,
    target = :out,
    args = (m[:in...]::NormalMeanVariance,),
    logscale = 0,
    body = (args) -> NormalMeanVariance(sum(mean, args.m[:in]), sum(var, args.m[:in])),
)
nothing # hide
```

The type after `m[:in...]` applies to each member. In the body, `args.m[:in]` is a tuple of the
members' messages, in member order. A call passes the group as a tuple too:

```@example groups
@call_message_update_rule(
    node = Sum, target = :out,
    m = (in = (NormalMeanVariance(1.0, 1.0), NormalMeanVariance(2.0, 0.5), NormalMeanVariance(-1.0, 2.0)),),
)
```

The card draws each member as an edge of its own, `in[1]` to `in[3]`. The same rule serves any
number of members, since the body sums over the tuple.

## A message towards a member

The message towards input ``x_k`` solves the sum for ``x_k``: ``x_k = y - \sum_{j \ne k} x_j``.
It takes the message on the output, ``\mathcal{N}(y \mid m_y, v_y)``, and the messages on the
other inputs:

```math
\mu_{f \to x_k}(x_k) = \int \delta\Big(y - \sum_j x_j\Big)\, \mathcal{N}(y \mid m_y, v_y) \prod_{j \ne k} \mathcal{N}(x_j \mid m_j, v_j)\, \mathrm{d}y\, \mathrm{d}x_{\setminus k}
                     = \mathcal{N}\Big(x_k \;\Big|\; m_y - \sum_{j \ne k} m_j,\ v_y + \sum_{j \ne k} v_j\Big).
```

One rule covers every member. Its target is `(:in, k)`, which binds `k` to the member's index,
and `m[:in][!k]` reads every member but `k`:

```@example groups
@define_message_update_rule(
    node = Sum,
    target = (:in, k),
    args = (m[:out]::NormalMeanVariance, m[:in][!k]::NormalMeanVariance),
    logscale = 0,
    body = (args) -> begin
        others = (m for (j, m) in enumerate(args.m[:in]) if j != k)
        NormalMeanVariance(mean(args.m[:out]) - sum(mean, others), var(args.m[:out]) + sum(var, others))
    end,
)
nothing # hide
```

The body sees `k` too. The group still arrives as a tuple of all its members, in member order,
with `nothing` at every position the selection leaves out, here position `k`. So
`args.m[:in][j]` is always member `j`, and the body skips `k` by its index. A call follows the
same layout, `nothing` in the target's place:

```@example groups
@call_message_update_rule(
    node = Sum, target = (:in, 2),
    m = (out = NormalMeanVariance(3.0, 0.1), in = (NormalMeanVariance(1.0, 1.0), nothing, NormalMeanVariance(-1.0, 2.0))),
)
```

The mean is ``3 - (1 - 1) = 3`` and the variance ``0.1 + 1 + 2 = 3.1``. The card draws the
target member as the arrow out.

A group input has one more form, which [Defining rules](@ref) lists with the others:
`m[:in][k]`, the target's own member, for a rule that needs the message on its own edge. The
tuple then holds member `k` and `nothing` elsewhere. Only a declared dependency delivers that
message, as [A node with its own algorithm](@ref tutorial-algorithm) shows.

## The log scales of the messages

Both rules declare `logscale = 0`. Each message is a convolution of normal densities, and a
convolution of normalised densities integrates to one, so the normalising constant is 1 and its
logarithm is 0. [Log scales](@ref) explains what the constant means in a graph.

## Clusters of a deterministic node

A [cluster](@ref glossary-cluster) is a set of interfaces whose variables share one factor of the
approximate posterior. A stochastic node's clusters follow the graph's
[factorisation](@ref glossary-factorisation). A deterministic node's clusters do not: they are
always the output alone and the joint over all its inputs. The factorisation cannot split the
inputs, because the output is a function of them.

| node | clusters |
|---|---|
| stochastic, `q(y) q(x₁) q(x₂)` | `(out)`, `(in[1])`, `(in[2])` |
| deterministic, any factorisation | `(out)`, `(in[1], in[2], …)` |

Belief propagation through a deterministic node reads the messages on every other interface, as
the two rules above do. The clusters matter for the
[Bethe free energy](@ref glossary-bethe-free-energy). A deterministic node needs no average
energy: its term is minus the entropy of the joint marginal over its inputs, so the node needs a
rule for that joint instead.

## The joint marginal of the inputs

The joint of the inputs is the product of every message on the node, restricted to where the sum
holds. With ``\mathbf{m}`` the vector of input means, ``\mathbf{v}`` the vector of input variances
and ``D = \operatorname{diag}(\mathbf{v})``:

```math
q(x) \propto \mathcal{N}\Big(\sum_k x_k \;\Big|\; m_y, v_y\Big) \prod_k \mathcal{N}(x_k \mid m_k, v_k)
     \propto \mathcal{N}\big(x \mid \mathbf{m} + \mathbf{g}\,(m_y - \textstyle\sum_k m_k),\ D - \mathbf{g}\mathbf{v}^\top\big),
\qquad \mathbf{g} = \frac{\mathbf{v}}{\sum_k v_k + v_y}.
```

This is a Kalman update of the inputs by one observation of their sum.
[`@define_marginal_update_rule`](@ref) defines the rule, and
[`@call_marginal_update_rule`](@ref) calls it. A marginal rule's target is a cluster, and
`(:in,)` names the joint over the group:

```@example groups
@define_marginal_update_rule(
    node = Sum,
    target = (:in,),
    args = (m[:out]::NormalMeanVariance, m[:in...]::NormalMeanVariance),
    body = (args) -> begin
        m, v = collect(mean.(args.m[:in])), collect(var.(args.m[:in]))
        g = v ./ (sum(v) + var(args.m[:out]))
        MvNormalMeanCovariance(m .+ g .* (mean(args.m[:out]) - sum(m)), Diagonal(v) - g * v')
    end,
)

@call_marginal_update_rule(
    node = Sum, target = (:in,),
    m = (out = NormalMeanVariance(4.0, 0.1), in = (NormalMeanVariance(1.0, 1.0), NormalMeanVariance(2.0, 0.5))),
)
```

The message on `out` says the sum is near 4, while the input messages put it at 3. The joint
moves both means up, the one with the larger variance further, and correlates the inputs
negatively: if one is larger, the other must be smaller.

## What the node can compute

[`rule_coverage`](@ref) shows a row per target: the output, any member of the group, and the
joint over the inputs.

```@example groups
MessagePassingRulesBase.rule_coverage(Sum)
```

[`check_rules`](@ref) finds nothing to report: every rule names interfaces the node declares, in
the forms its groups allow.

```@example groups
MessagePassingRulesBase.check_rules(@__MODULE__)
```

The rules accept only `NormalMeanVariance`. A message of another type, such as an observation of
the output, finds no rule, and the error shows which input did not fit:

```@example groups
try
    @call_message_update_rule(
        node = Sum, target = (:in, 1),
        m = (out = PointMass(3.0), in = (nothing, NormalMeanVariance(1.0, 1.0))),
    )
catch err
    showerror(stdout, err)
end
```

## Next steps

- [Your first node](@ref tutorial-first-node) covers the basics this tutorial builds on.
- [A node with its own algorithm](@ref tutorial-algorithm) gives a node a parametrised algorithm
  and declares what its rules take.
- [Defining nodes](@ref) describes groups and what a node requires of a graph, and
  [Defining rules](@ref) every input form.
- [Algorithms and dependencies](@ref) explains which inputs a rule receives.
- The [Keyword reference](@ref keyword-reference) lists every keyword of every macro.
