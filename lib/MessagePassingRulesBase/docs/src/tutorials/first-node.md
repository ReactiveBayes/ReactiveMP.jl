```@meta
CurrentModule = MessagePassingRulesBase
```

# [Your first node](@id tutorial-first-node)

This tutorial builds a [factor node](@ref glossary-factor-node) from nothing: a normal
distribution with a known variance,

```math
f(y, x, v) = \mathcal{N}(y \mid x, v),
```

where `y` is the output, `x` the mean and `v` the variance. You declare the node, write its
[belief propagation](@ref glossary-belief-propagation) and
[variational](@ref glossary-vmp) rules, give it an
[average energy](@ref glossary-average-energy), and check what it can compute. Every step runs
the rules by hand, as a test would. An engine such as
[ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) runs the same rules in a
graph.

The messages are distributions from
[ExponentialFamily](https://github.com/ReactiveBayes/ExponentialFamily.jl), and observations and
constants arrive as a [`PointMass`](@ref glossary-point-mass) from BayesBase.

```@example first-node
using MessagePassingRulesBase, BayesBase, ExponentialFamily
nothing # hide
```

## Declare the node

A node is a type, here an empty `struct`, and a declaration that names its interfaces. The
first interface is the output. The mean also answers to `mean`, an alias.

```@example first-node
struct Gaussian end

@define_factor_node(
    node = Gaussian,
    type = Stochastic,
    interfaces = [:out, (:μ, aliases = [:mean]), :v],
)
```

[`Stochastic`](@ref) says the node is a density over its interfaces, as opposed to a
[`Deterministic`](@ref) function. The declaration is data, which [`nodespec`](@ref) returns and
which draws itself:

```@example first-node
MessagePassingRulesBase.nodespec(Gaussian)
```

The node has no rules yet, so it can compute nothing.

## A message towards the output

The message towards `out` says what the node knows about `y` from the messages on its other
edges. For a normal message on the mean, ``\mathcal{N}(x \mid m, s)``, and a known variance
``v``, belief propagation integrates the mean out:

```math
\mu_{f \to y}(y) = \int \mathcal{N}(y \mid x, v)\, \mathcal{N}(x \mid m, s)\, \mathrm{d}x
                 = \mathcal{N}(y \mid m, s + v).
```

The rule says exactly that. Its target is `out`; its inputs are the message on `μ`, a
`NormalMeanVariance`, and the message on `v`, a point mass; its body computes the result from
them:

```@example first-node
@define_message_update_rule(
    node = Gaussian,
    target = :out,
    args = (m[:μ]::NormalMeanVariance, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), var(args.m[:μ]) + mean(args.m[:v])),
)
nothing # hide
```

`m[:μ]` reads as "the message on `μ`". The body receives the inputs as `args`, and
`args.m[:μ]` is that message. The `logscale` keyword states the logarithm of the message's
normalising constant, which the next-but-one section explains.

Call the rule with inputs of your choice, as an engine would:

```@example first-node
@call_message_update_rule(
    node = Gaussian, target = :out,
    m = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),
)
```

The result draws the node with the edges the rule read, messages as solid arrows in, and the
target as the arrow out. Its value is `NormalMeanVariance(1.0, 2.5)`: the variances add.

## A message towards the mean

The message towards `μ` is the same integral taken the other way. When `y` is observed, its
message is a point mass at the observation, and the message towards `μ` is the likelihood of
the observation as a function of the mean, ``\mathcal{N}(y \mid x, v)``, a normal in ``x``:

```@example first-node
@define_message_update_rule(
    node = Gaussian,
    target = :μ,
    args = (m[:out]::PointMass, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:out]), mean(args.m[:v])),
)

@define_message_update_rule(
    node = Gaussian,
    target = :μ,
    args = (m[:out]::NormalMeanVariance, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:out]), var(args.m[:out]) + mean(args.m[:v])),
)

@call_message_update_rule(node = Gaussian, target = :μ, m = (out = PointMass(3.0), v = PointMass(0.5)))
```

Two rules share the target `μ`. The types of the inputs select between them, as Julia's
dispatch selects a method; the card lists the other rule for the same target and shows why it
does not fit this call.

## Which inputs a rule takes

The rules so far take messages, because every interface of the node is in one
[cluster](@ref glossary-cluster). Which inputs a rule receives is not the rule's choice. It
follows from the [factorisation](@ref glossary-factorisation) of the approximate posterior: a
rule takes the **messages** on the other interfaces of its target's cluster, and the
**marginals** of the other clusters.

| factorisation | clusters | the rule towards `out` takes | which is |
|---|---|---|---|
| `q(y, x, v)` | `(out, μ, v)` | `m[:μ]`, `m[:v]` | belief propagation |
| `q(y) q(x) q(v)` | `(out)`, `(μ)`, `(v)` | `q[:μ]`, `q[:v]` | [mean-field](@ref glossary-mean-field) variational message passing |
| `q(y, x) q(v)` | `(out, μ)`, `(v)` | `m[:μ]`, `q[:v]` | [structured](@ref glossary-structured-vmp) variational message passing |

So a node supports a factorisation when it has rules for the inputs that factorisation
delivers. [Algorithms and dependencies](@ref) describes this scheme in full.

## A variational rule

Under the mean-field factorisation, the rule towards `out` takes the [marginals](@ref glossary-marginal)
`q(x)` and `q(v)`. Variational message passing sends the exponentiated expected log-density:

```math
\mu_{f \to y}(y) \propto \exp \mathbb{E}_{q(x)}\big[\log \mathcal{N}(y \mid x, v)\big]
                 \propto \mathcal{N}\big(y \mid \mathbb{E}[x], v\big).
```

Only the mean of `q(x)` matters, so the rule accepts any marginal with a mean:

```@example first-node
@define_message_update_rule(
    node = Gaussian,
    target = :out,
    args = (q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> NormalMeanVariance(mean(args.q[:μ]), mean(args.q[:v])),
)

@call_message_update_rule(
    node = Gaussian, target = :out,
    q = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),
)
```

The inputs are drawn dashed, as marginals, and the variance of `q(x)` no longer reaches the
result. The rule declares no `logscale`, so the result's log scale is undefined: an expected
log-density has no normalising constant with a meaning of its own.

## The log scale of a message

A message is a distribution up to a constant, and the [log scale](@ref glossary-log-scale) is
the logarithm of that constant. The belief propagation rules above declare `logscale = 0`
because their integrals are already normalised: a convolution of normals integrates to one.
Summed along a graph, log scales give the model's evidence, so a rule that declares one must
state it correctly. [Log scales](@ref) covers the other ways to declare one.

## The average energy

The [Bethe free energy](@ref glossary-bethe-free-energy), the quantity message passing
minimises, needs each node's average energy, its expected negative log-density under the
marginals of its clusters. Under the mean-field factorisation with a known variance:

```math
U = \tfrac{1}{2}\log(2\pi v) + \frac{\operatorname{var}[y] + \operatorname{var}[x] + (\mathbb{E}[y] - \mathbb{E}[x])^2}{2v}.
```

```@example first-node
@define_average_energy(
    node = Gaussian,
    args = (q[:out]::Any, q[:μ]::Any, q[:v]::PointMass),
    body = (args) -> begin
        y, x, v = args.q[:out], args.q[:μ], mean(args.q[:v])
        (log(2v * π) + (var(y) + var(x) + (mean(y) - mean(x))^2) / v) / 2
    end,
)

@call_average_energy(
    node = Gaussian,
    q = (out = NormalMeanVariance(0.0, 1.0), μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)),
)
```

## What the node can compute

[`rule_coverage`](@ref) tabulates the node's rules: a row per target and one for the average
energy, a column per algorithm.

```@example first-node
MessagePassingRulesBase.rule_coverage(Gaussian)
```

[`which_message_update_rule`](@ref) finds the rule a call would run, without running it, and
shows what it consumes:

```@example first-node
which_message_update_rule(Gaussian, :out; q = (μ = NormalMeanVariance(1.0, 2.0), v = PointMass(0.5)))
```

[`check_rules`](@ref) compares every rule with its node's declaration, and returns the problems
it finds; a rule package's tests run it.

```@example first-node
MessagePassingRulesBase.check_rules(@__MODULE__)
```

## When no rule fits

The node has no rule towards `v`. Asking for one throws a [`RuleNotFoundError`](@ref), which
says what was asked and, for every rule of the node and target, why it does not fit:

```@example first-node
try
    @call_message_update_rule(node = Gaussian, target = :v, m = (out = PointMass(3.0), μ = PointMass(1.0)))
catch err
    showerror(stdout, err)
end
```

The same error reaches a model's user when a graph needs a rule that no package defines.

## Next steps

- [A deterministic node with a group](@ref tutorial-groups) writes rules for `out = in₁ + in₂ + …`.
- [A node with its own algorithm](@ref tutorial-algorithm) gives a node a parametrised algorithm
  and declares what its rules take.
- [Defining nodes](@ref), [Defining rules](@ref) and the [Keyword reference](@ref keyword-reference)
  cover every option.
- To use a node in a model, see [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/)
  and [RxInfer](https://reactivebayes.github.io/RxInfer.jl/stable/).
