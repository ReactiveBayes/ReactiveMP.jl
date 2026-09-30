```@meta
CurrentModule = MessagePassingRulesBase
```

# [A node with its own algorithm](@id tutorial-algorithm)

This tutorial builds a logistic observation node, a binary output whose probability is the
logistic function of a real input:

```math
f(y, x) = \operatorname{Bernoulli}\big(y \mid \sigma(x)\big), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}.
```

Its messages have no closed form, so the rules compute them numerically, with a number of points
you choose. That number is a parameter of the rules, not an input, and it lives in the node's own
[algorithm](@ref glossary-algorithm). The rule towards `x` is an
[expectation propagation](@ref glossary-expectation-propagation) rule: it reads the message on
its own edge, which the [default scheme](@ref glossary-default-scheme) never delivers, so the
node declares what its rules take. The tutorial assumes you have read
[Your first node](@ref tutorial-first-node).

```@example algorithm
using MessagePassingRulesBase, BayesBase, ExponentialFamily
nothing # hide
```

## The algorithm

An algorithm is a type, and its fields are the rules' parameters. This one carries the number of
quadrature points:

```@example algorithm
struct LogisticQuadrature <: AbstractAlgorithm
    n::Int
end
```

A direct subtype of [`AbstractAlgorithm`](@ref) stands alone: only the rules and dependencies
declared for it apply. A subtype of [`DefaultAlgorithmExtension`](@ref) instead overrides some
rules of a node that runs under [`DefaultAlgorithm`](@ref) and inherits the rest. The logistic
node has no default rules to inherit, so its algorithm stands alone.

## Declare the node

[`@define_factor_node`](@ref) declares the node. Its [`algorithm`](@ref keyword-node-algorithm)
keyword makes `LogisticQuadrature(32)` the node's default algorithm: the one its rules run under
unless a call or a graph asks for another. The section on the initial message below explains
[`initial_messages`](@ref keyword-node-initial_messages).

```@example algorithm
struct Logistic end

@define_factor_node(
    node = Logistic,
    type = Stochastic,
    interfaces = [:out, :in],
    algorithm = LogisticQuadrature(32),
    initial_messages = [:in => NormalMeanVariance(0.0, 100.0)],
)

MessagePassingRulesBase.nodespec(Logistic)
```

The keyword takes a value, as here, or a type, which the macro instantiates with no arguments.
[`default_algorithm`](@ref) returns the value, and a rule that omits `algorithm` belongs to its
type, `LogisticQuadrature`.

## What the rules take

Under the default scheme, the [factorisation](@ref glossary-factorisation) decides a rule's
inputs: the messages on the other interfaces of the target's [cluster](@ref glossary-cluster) and
the marginals of the other clusters. For the rule towards `in`:

| factorisation | the default scheme gives the rule towards `in` |
|---|---|
| `q(y, x)` | `m[:out]` |
| `q(y) q(x)` | `q[:out]` |

Neither gives `m[:in]`: under the default scheme, a rule never receives the message on its own
edge. The expectation propagation rule needs it, whatever the factorisation. So the node declares
its [dependencies](@ref glossary-dependencies), the inputs each target's rule takes, for its
algorithm. Until it does, [`dependencies_spec`](@ref) returns `nothing`, and an engine follows the
default scheme:

```@example algorithm
MessagePassingRulesBase.dependencies_spec(Logistic, LogisticQuadrature(32)) === nothing
```

[`@define_dependencies`](@ref) declares them, target by target, in the same notation as a rule's
`args` without the types:

```@example algorithm
@define_dependencies(
    node = Logistic,
    algorithm = LogisticQuadrature,
    dependencies = [:out => (m[:in],), :in => (m[:out], m[:in])],
)

MessagePassingRulesBase.dependencies_spec(Logistic, LogisticQuadrature(32))
```

The declaration draws the node with each target's inputs. Three things hold for every
declaration:

- Every target a graph connects must be declared. A target that follows the default scheme is
  written `target => (default,)`. `:a => (default, q[:a])` adds one input to the default scheme's.
- The inputs are subscribed to in the order written. Under variational message passing, that
  order is the update schedule.
- The declaration is checked against the node's interfaces when it loads, so an unknown name is
  an error at once.

[`@define_factor_node`](@ref) takes the same list in its
[`dependencies`](@ref keyword-node-dependencies) keyword, for the node's default algorithm.
[`@define_dependencies`](@ref) also works after the declaration, and for any other algorithm
of the node.

## A numerical expectation

Both rules need expectations of a function under a normal distribution,
``\mathbb{E}[g(x)]`` for ``x \sim \mathcal{N}(m, v)``. The helper evaluates ``g`` at ``n``
evenly spaced points within six standard deviations of the mean, weighted by the normal density:

```@example algorithm
σ(x) = 1 / (1 + exp(-x))

function gaussian_expectation(g, m, v, n)
    z = range(-6, 6; length = n)
    w = exp.(-z .^ 2 ./ 2)
    return sum(w .* g.(m .+ sqrt(v) .* z)) / sum(w)
end
nothing # hide
```

More points give a more accurate expectation, at a higher cost. The rules share this helper
instead of calling each other.

## A message towards the output

The message towards `y` integrates the input out, and the result is a Bernoulli distribution:

```math
\mu_{f \to y}(y) = \int \operatorname{Bernoulli}\big(y \mid \sigma(x)\big)\, \mathcal{N}(x \mid m, v)\, \mathrm{d}x
                 = \operatorname{Bernoulli}\big(y \mid \mathbb{E}[\sigma(x)]\big).
```

The rule's [`body`](@ref keyword-message-body) names the slot `algo` before `args`, and reads
the number of points from it:

```@example algorithm
@define_message_update_rule(
    node = Logistic,
    target = :out,
    args = (m[:in]::NormalMeanVariance,),
    logscale = 0,
    body = (algo, args) -> Bernoulli(gaussian_expectation(σ, mean(args.m[:in]), var(args.m[:in]), algo.n)),
)

@call_message_update_rule(node = Logistic, target = :out, m = (in = NormalMeanVariance(1.0, 4.0),))
```

The call gives no algorithm, so it runs under the node's default, `LogisticQuadrature(32)`. The
log scale is 0 because a Bernoulli distribution is normalised: the message sums to one over
``y``, however accurate the quadrature. A call may pass another algorithm value:

```@example algorithm
@call_message_update_rule(
    node = Logistic, target = :out,
    m = (in = NormalMeanVariance(1.0, 4.0),), algorithm = LogisticQuadrature(3),
)
```

Three points are too few: the probability moves from about 0.648 to 0.731. Under
`DefaultAlgorithm`, no rule of the node applies, and the error says which algorithm the rules
belong to:

```@example algorithm
try
    @call_message_update_rule(
        node = Logistic, target = :out,
        m = (in = NormalMeanVariance(1.0, 4.0),), algorithm = DefaultAlgorithm(),
    )
catch err
    showerror(stdout, err)
end
```

## A message towards the input

The exact message towards ``x`` is the likelihood of ``y``, ``\sigma(x)^y (1 - \sigma(x))^{1-y}``,
which is not a normal density. Expectation propagation approximates it by a normal one. The
message on the node's own edge, ``\mathcal{N}(x \mid m, v)``, is the cavity: what the rest of the
graph says about ``x``. Its product with the likelihood is the tilted distribution, and the rule
matches its mean ``\hat{m}`` and variance ``\hat{v}``:

```math
\tilde{p}(x) \propto \sigma(x)^y (1 - \sigma(x))^{1-y}\, \mathcal{N}(x \mid m, v), \qquad
\mu_{f \to x}(x) \propto \frac{\mathcal{N}(x \mid \hat{m}, \hat{v})}{\mathcal{N}(x \mid m, v)}.
```

A ratio of normal densities is a normal density with precision ``1/\hat{v} - 1/v`` and weighted
mean ``\hat{m}/\hat{v} - m/v``:

```@example algorithm
@define_message_update_rule(
    node = Logistic,
    target = :in,
    args = (m[:out]::PointMass, m[:in]::NormalMeanVariance),
    body = (algo, args) -> begin
        y, m, v = mean(args.m[:out]), mean(args.m[:in]), var(args.m[:in])
        likelihood(x) = y == 1 ? σ(x) : 1 - σ(x)
        Z = gaussian_expectation(likelihood, m, v, algo.n)
        m̂ = gaussian_expectation(x -> x * likelihood(x), m, v, algo.n) / Z
        v̂ = gaussian_expectation(x -> (x - m̂)^2 * likelihood(x), m, v, algo.n) / Z
        NormalWeightedMeanPrecision(m̂ / v̂ - m / v, 1 / v̂ - 1 / v)
    end,
)

@call_message_update_rule(
    node = Logistic, target = :in,
    m = (out = PointMass(1.0), in = NormalMeanVariance(0.0, 4.0)),
)
```

The card draws `in` as the target, and its rule section lists both inputs the rule declares. It
labels the rule belief propagation because the rule reads only messages; the algorithm makes it
expectation propagation. The rule declares no `logscale`, so the result's
[log scale](@ref glossary-log-scale) is undefined.

## An initial message on the input

The rule towards `in` waits for the message on `in`. Suppose `x` is a weight shared by several
logistic nodes, as in logistic regression. The message arriving at one node's `in` is the product
of the messages the other nodes send towards `x`, and each of those waits for its own message on
`in`. No rule can run first.

An [initial message](@ref glossary-initial-message) breaks the cycle. An engine places it on the
node's inbound edge before inference, where the model sets nothing; a model's own initialisation
wins. [`initial_messages`](@ref) reads the node's:

```@example algorithm
MessagePassingRulesBase.initial_messages(Logistic)
```

A broad normal suits a starting point: it lets the first update depend on the data rather than on
the initial message. The initial message is not a dependency, and which inputs a rule reads stays
the algorithm's. No rule computed it, so its log scale is undefined.

## A parametric algorithm

`LogisticQuadrature` has no type parameters. Suppose it had one, `LogisticQuadrature{T}` with a
field `n::T`. The node's default, `LogisticQuadrature(32)`, is then a `LogisticQuadrature{Int}`,
and a rule that omits `algorithm` belongs to that type alone. A call with
`LogisticQuadrature(0x20)`, a `LogisticQuadrature{UInt8}`, would find no rule. Rules meant for
every variant declare `algorithm = LogisticQuadrature`, which matches every
`LogisticQuadrature{…}`. `@define_dependencies` with `algorithm = LogisticQuadrature`, as above,
already covers them all. The `dependencies` keyword of `@define_factor_node` does not: like a rule
that omits `algorithm`, it binds to the type of the default instance.

## What the node can compute

[`rule_coverage`](@ref) has one column, the node's algorithm:

```@example algorithm
MessagePassingRulesBase.rule_coverage(Logistic)
```

[`check_rules`](@ref) compares the rules with the node's declaration and its dependencies. Each
rule reads what its target is declared to take, so it reports nothing:

```@example algorithm
MessagePassingRulesBase.check_rules(@__MODULE__)
```

The node has no average energy, so an engine cannot compute a free energy with it.

## Next steps

- [Your first node](@ref tutorial-first-node) and
  [A deterministic node with a group](@ref tutorial-groups) cover rules under the default
  algorithm.
- [Algorithms and dependencies](@ref) describes the default scheme, extensions of the default and
  every form of a dependency.
- [Defining nodes](@ref) and [Defining rules](@ref) cover the other keywords of the node and of
  its rules.
- The [Keyword reference](@ref keyword-reference) lists every keyword of every macro.
