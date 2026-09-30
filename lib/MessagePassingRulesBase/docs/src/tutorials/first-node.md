```@meta
CurrentModule = MessagePassingRulesBase
```

# [Your first node](@id tutorial-first-node)

This tutorial builds a [factor node](@ref glossary-factor-node) from nothing: a normal
distribution with a known variance,

```math
f(y, x, v) = \mathcal{N}(y \mid x, v),
```

where `y` is the output, `x` the mean and `v` the variance. You define the distribution, declare
it as a node, write its [belief propagation](@ref glossary-belief-propagation) and
[variational](@ref glossary-vmp) rules, give it an
[average energy](@ref glossary-average-energy), and check what it can compute. Every step runs
the rules by hand, as a test would. An engine such as
[ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) runs the same rules in a
graph.

Observations and constants reach a rule as a [`PointMass`](@ref glossary-point-mass) from
BayesBase, and the distribution interface (`mean`, `var`, `logpdf`) comes from Distributions.

```@example first-node
using MessagePassingRulesBase, BayesBase, Distributions
nothing # hide
```

## What a node is

A node is a Julia value that names a factor of a model: a type, or a function such as `+`. The
node is what everything else refers to. A rule names it, `node = Gaussian`, and finding a rule is
Julia's dispatch on its type, so any loaded package can add rules for a node another package
declared ([Defining rules](@ref) explains how). This package builds no models: an engine such as
ReactiveMP places the node in a graph, and a model written with [RxInfer](https://reactivebayes.github.io/RxInfer.jl/stable/) names it in a statement such
as `y ~ Gaussian(x, v)`.

Most stochastic nodes are probability distributions, and the rule packages use the
distribution's own type as the node. StandardMessagePassingRules, for example, declares
ExponentialFamily's `NormalMeanVariance` as a node. One type then does three jobs:

- **it names the factor** in a model, `y ~ NormalMeanVariance(μ, v)` in RxInfer's syntax;
- **it is the factor's density**: the declaration derives the node's log-density from it,
  `logpdf(NormalMeanVariance(μ, v), y)`, which rule fallbacks and rule tests use;
- **it is often a message**: the message towards `out` from point-mass inputs is the node's own
  density, `NormalMeanVariance(μ, v)`, and many other rules return the same family.

Not every message has the node's type: the message towards a variance is another family. Nor is
every node a distribution. A deterministic function is a node of its own, as
[A deterministic node with a group](@ref tutorial-groups) shows with `+`, and a factor that has
no distribution type is named by an empty type, `struct MyFactor end`, used only as a name.

This tutorial follows the pattern with a small normal distribution of its own. A real package
takes the type from ExponentialFamily, which also defines products and conversions of messages.

```@example first-node
struct Gaussian{T <: Real} <: ContinuousUnivariateDistribution
    μ::T   # the mean
    v::T   # the variance
end

Gaussian(μ::Real, v::Real) = Gaussian(promote(μ, v)...)

Distributions.mean(d::Gaussian) = d.μ
Distributions.var(d::Gaussian) = d.v
Distributions.logpdf(d::Gaussian, x::Real) = -(log(2π * d.v) + abs2(x - d.μ) / d.v) / 2
nothing # hide
```

## Declare the node

[`@define_factor_node`](@ref) declares the type as a node. The declaration names the node's
[interfaces](@ref glossary-interface), its edges in a graph, in order: the output first, then the
distribution's parameters in the order its constructor takes them. The mean also answers to
`mean`, an alias.

```@example first-node
@define_factor_node(
    node = Gaussian,
    type = Stochastic,
    interfaces = [:out, (:μ, aliases = [:mean]), :v],
)
```

Each keyword has a section in the [Keyword reference](@ref keyword-factor-node):
[`node`](@ref keyword-node-node), [`type`](@ref keyword-node-type) and
[`interfaces`](@ref keyword-node-interfaces), and the optional ones this node does not need.
[`Stochastic`](@ref) says the node is a density over its interfaces, as opposed to a
[`Deterministic`](@ref) function. The declaration is data, which [`nodespec`](@ref) returns and
which draws itself:

```@example first-node
MessagePassingRulesBase.nodespec(Gaussian)
```

Because the node is a distribution, callable with its parameters, the declaration also gives its
log-density as a function of the interfaces, [`nodefunction`](@ref):

```@example first-node
f = MessagePassingRulesBase.nodefunction(Gaussian)
f(out = 1.0, μ = 0.0, v = 2.0) ≈ logpdf(Normal(0.0, sqrt(2.0)), 1.0)
```

The node has no rules yet, so it can compute no message. [Defining nodes](@ref) covers every
part of a declaration.

## A message towards the output

The message towards `out` says what the node knows about `y` from the messages on its other
edges. For a normal message on the mean, ``\mathcal{N}(x \mid m, s)``, and a known variance
``v``, belief propagation integrates the mean out:

```math
\mu_{f \to y}(y) = \int \mathcal{N}(y \mid x, v)\, \mathcal{N}(x \mid m, s)\, \mathrm{d}x
                 = \mathcal{N}(y \mid m, s + v).
```

[`@define_message_update_rule`](@ref) defines the rule. Its
[`target`](@ref keyword-message-target) is `out`; its [`args`](@ref keyword-message-args) are
the message on `μ`, a `Gaussian` or a point mass, and the message on `v`, a point mass; its
[`body`](@ref keyword-message-body) computes the result from them. The result is the node's own
type:

```@example first-node
@define_message_update_rule(
    node = Gaussian,
    target = :out,
    args = (m[:μ]::Union{Gaussian, PointMass}, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> Gaussian(mean(args.m[:μ]), var(args.m[:μ]) + mean(args.m[:v])),
)
nothing # hide
```

`m[:μ]` reads as "the message on `μ`". The body receives the inputs as `args`, and
`args.m[:μ]` is that message; a point mass has variance zero, so one body serves both. The
[`logscale`](@ref keyword-message-logscale) keyword states the logarithm of the message's
normalising constant, which [The log scale of a message](@ref tutorial-first-node-logscale)
explains.

[`@call_message_update_rule`](@ref) calls the rule with inputs of your choice, as an engine
would:

```@example first-node
@call_message_update_rule(
    node = Gaussian, target = :out,
    m = (μ = Gaussian(1.0, 2.0), v = PointMass(0.5)),
)
```

The result draws the node with the edges the rule read, messages as solid arrows in, and the
target as the arrow out. Its value is `Gaussian(1.0, 2.5)`: the variances add. With the mean
observed too, the message is the node's density itself:

```@example first-node
@call_message_update_rule(node = Gaussian, target = :out, m = (μ = PointMass(1.0), v = PointMass(0.5)))
```

## A message towards the mean

The message towards `μ` is the same integral taken the other way. When `y` is observed, its
message is a point mass at the observation, and the message towards `μ` is the likelihood of
the observation as a function of the mean, ``\mathcal{N}(y \mid x, v)``, a normal in ``x``. The
density is symmetric in `y` and `x`, so the rule mirrors the one towards `out`:

```@example first-node
@define_message_update_rule(
    node = Gaussian,
    target = :μ,
    args = (m[:out]::Union{Gaussian, PointMass}, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> Gaussian(mean(args.m[:out]), var(args.m[:out]) + mean(args.m[:v])),
)

@call_message_update_rule(node = Gaussian, target = :μ, m = (out = PointMass(3.0), v = PointMass(0.5)))
```

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
`q(x)` and `q(v)`, written `q[:μ]` and `q[:v]` among its `args`. Variational message passing
sends the exponentiated expected log-density:

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
    body = (args) -> Gaussian(mean(args.q[:μ]), mean(args.q[:v])),
)

@call_message_update_rule(
    node = Gaussian, target = :out,
    q = (μ = Gaussian(1.0, 2.0), v = PointMass(0.5)),
)
```

The inputs are drawn dashed, as marginals, and the variance of `q(x)` no longer reaches the
result. The rule declares no `logscale`, so the result's log scale is undefined: an expected
log-density has no normalising constant with a meaning of its own.

## [The log scale of a message](@id tutorial-first-node-logscale)

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

[`@define_average_energy`](@ref) defines it, with the same [`args`](@ref keyword-energy-args)
and [`body`](@ref keyword-energy-body) as a rule and no target, and
[`@call_average_energy`](@ref) calls it:

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
    q = (out = Gaussian(0.0, 1.0), μ = Gaussian(1.0, 2.0), v = PointMass(0.5)),
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
which_message_update_rule(Gaussian, :out; q = (μ = Gaussian(1.0, 2.0), v = PointMass(0.5)))
```

[`check_rules`](@ref) compares every rule with its node's declaration, and returns the problems
it finds; a rule package's tests run it. [Inspecting rules](@ref) covers these queries.

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
[Calling rules](@ref) describes every part of the report.

## Next steps

- [A deterministic node with a group](@ref tutorial-groups) writes rules for `out = in₁ + in₂ + …`.
- [A node with its own algorithm](@ref tutorial-algorithm) gives a node a parametrised algorithm
  and declares what its rules take.
- [Defining nodes](@ref), [Defining rules](@ref) and the [Keyword reference](@ref keyword-reference)
  cover every option.
- To use a node in a model, see [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/)
  and [RxInfer](https://reactivebayes.github.io/RxInfer.jl/stable/).
