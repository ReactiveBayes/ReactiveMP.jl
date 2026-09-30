# [Log scales](@id lib-logscale)

A message's [log scale](@extref MessagePassingRulesBase glossary-log-scale) is a scalar carried
beside its distribution. A rule returns a normalised distribution, but what it computes may be an
unnormalised function. The log scale is the constant that relates the two:

```math
\text{message}(x) = \exp(\text{logscale}) \cdot \hat{p}(x),
```

where ``\hat{p}`` is the distribution the rule returns. A rule that can compute its log scale
declares it, and the engine carries it along every message and combines it through every product.
Where a rule cannot compute it, the log scale is *undefined*, and says why.

```@setup logscale
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, MessageProductContext, get_stream_of_marginals
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The examples use the `Gaussian` node of [The example node](@ref example-node).

## [The log evidence of a graph](@id lib-logscale-evidence)

The model below is a latent `x` with a normal prior, observed once through normal noise:

```math
x \sim \mathcal{N}(0, 10), \qquad y \mid x \sim \mathcal{N}(x, 1).
```

Its evidence is ``p(y) = \mathcal{N}(y \mid 0, 11)``. Every node is activated with
`logscales = true`:

```@example logscale
function posterior(factorisation; observation = 2.0)
    x, y = randomvar(label = :x), datavar(label = :y)
    prior = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
    likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constvar(1.0))], factorisation)
    activate!(x, RandomVariableActivationOptions())
    activate!(y, DataVariableActivationOptions())
    for node in (prior, likelihood)
        activate!(node, FactorNodeActivationOptions(; logscales = true))
    end
    posteriors = Marginal[]
    subscription = subscribe!(get_stream_of_marginals(x), (q) -> push!(posteriors, q))
    new_observation!(y, observation)
    unsubscribe!(subscription)
    return last(posteriors)
end

q = posterior(nothing)
getlogscale(q), logpdf(Normal(0.0, sqrt(11.0)), 2.0)
```

Both nodes run [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation),
and [`getlogscale`](@ref getlogscale(::Marginal)) of the posterior is the log evidence.

The same graph with a [mean-field](@extref MessagePassingRulesBase glossary-mean-field)
likelihood runs a variational rule towards `x`. The posterior is the same, but its log scale is
undefined, and says which rule made it so:

```@example logscale
getlogscale(posterior(((:out,), (:μ,), (:v,))))
```

## Why it matters: evidence in message passing

A belief propagation message ``\mu(x) = \int f(x, y, \dots) \prod_i \mu_i(y_i) \,\mathrm{d}y`` is
in general not normalised. In sum-product message passing on a Forney-style factor graph, a
message ``\vec{\mu}_{s_j}(s_j)`` along an edge decomposes as
``\beta_{s_j} \cdot \hat{p}_{s_j}(s_j)``: the normalised distribution ReactiveMP stores as the
message's data, and a positive **scale factor**. A key result of
[van Erp et al. (2023)](https://arxiv.org/abs/2306.05965) is that in an acyclic graph the
product of two colliding messages integrates to the model evidence:

```math
\int \vec{\mu}_{s_j}(s_j)\, \overleftarrow{\mu}_{s_j}(s_j)\, \mathrm{d}s_j = p(y = \hat{y}).
```

So you can read the evidence off locally at *any* edge, from the log scale of a variable's
marginal, in the same run that computes the posteriors. This makes **Bayesian model comparison**
(averaging, selection and combination) part of inference. The `Mixture` node of
[StandardMessagePassingRules](@extref StandardMessagePassingRules StandardMessagePassingRules)
reads the log scales of its inputs, and its message towards the switch is a softmax over the
evidence of each component. In a streaming model, the change of a posterior's log scale from one
observation to the next is the one-step predictive likelihood.

Nothing restricts log scales to belief propagation: any rule whose result stands for an
unnormalised function may declare its constant. A naive variational message,
``\exp \mathbb{E}_q[\log f]``, has no constant with that meaning, so its log scale is undefined.

## How a product combines them

ReactiveMP tracks the logarithm of the scale factor. When two messages are multiplied, the log
scale of the product is

```math
\log \beta_\text{new} = \log \beta_\text{left} + \log \beta_\text{right} + \log \int \hat{p}_\text{left}(x)\, \hat{p}_\text{right}(x)\, \mathrm{d}x.
```

The last term is `BayesBase.compute_logscale(new, left, right)`. The product of two normalised
Gaussians is not normalised,
``\mathcal{N}(x \mid a, A)\, \mathcal{N}(x \mid b, B) = \mathcal{N}(a \mid b, A + B)\, \mathcal{N}(x \mid c, C)``,
and ``\log \mathcal{N}(a \mid b, A + B)`` is that term:

```@example logscale
left = Message(NormalMeanVariance(0.0, 10.0), false, false, ReactiveMP.AnnotationDict(), 0.0)
right = Message(NormalMeanVariance(2.0, 1.0), false, false, ReactiveMP.AnnotationDict(), 0.0)
product = ReactiveMP.compute_product_of_two_messages(randomvar(), MessageProductContext(), left, right)
getlogscale(product), logpdf(Normal(2.0, sqrt(11.0)), 0.0)
```

These are the two messages that meet at `x` in the first example. In a Gaussian model, most of
the evidence accumulates in the products rather than in the rules.

## What a rule declares

The normalised distribution a rule returns does not reveal its log scale: `Beta(2, 1)` looks the
same whether or not a constant was divided out to get it. So a rule states it, with the
`logscale` keyword of
[`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule):

- a constant, such as `logscale = 0` for a convolution of Gaussians;
- a function of the inputs;
- [`from_body`](@extref MessagePassingRulesBase.from_body), with the body returning
  [`with_logscale`](@extref MessagePassingRulesBase.with_logscale)`(result, logscale)`.

The rule of `Gaussian` towards `out` declares `logscale = 0`, which its result shows:

```@example logscale
@call_message_update_rule(node = Gaussian, target = :out, m = (μ = NormalMeanVariance(0.0, 10.0), v = PointMass(1.0)))
```

A rule that omits the keyword has an
[`UndefinedLogScale`](@extref MessagePassingRulesBase.UndefinedLogScale) that names the rule.
Every variational rule is such a rule:

```@example logscale
result = @call_message_update_rule(node = Gaussian, target = :μ, q = (out = PointMass(2.0), v = PointMass(1.0)))
getlogscale(result)
```

A rule that needs the log scales its inputs arrived with declares `reads_logscale = true` and
reads them as `args.logscale.m[:out]`.
[`require_logscale`](@extref MessagePassingRulesBase.require_logscale) turns an undefined one into
an error that says why, as the `Mixture` rules do:

```@example logscale
try
    require_logscale(getlogscale(result))
catch err
    showerror(stdout, err)
end
```

How rules declare and test their log scales is the subject of
[Log scales](@extref MessagePassingRulesBase Log-scales) in the base package.

## What the engine does

The engine tracks log scales when you activate the nodes with `logscales = true`
([`ReactiveMP.FactorNodeActivationOptions`](@ref); RxInfer's `infer(...; logscales = true)`).
Then:

- every message carries the log scale its rule declares, and a message a rule fallback computed
  carries an undefined one;
- observations and constants are point masses, with log scale zero;
- initial messages were computed by no rule, so their log scale is undefined;
- a product combines the log scales as above. An undefined side propagates, keeping its reason.
  A pair of distributions with no `compute_logscale` gives an undefined log scale, and so does a
  form constraint that changes the product. A `missing` side leaves the other side's log scale;
- a random variable's marginal carries the log scale of the product it was formed from;
- an initial marginal, set with [`ReactiveMP.set_initial_marginal!`](@ref), and a factor node's
  joint marginal carry none: reading theirs is an error, as where log scales are not tracked.

Nothing errors because a log scale is undefined, except a rule or a user that asks for its value.
Without `logscales = true`, messages carry `nothing`, nothing is computed, and a rule declared
with `reads_logscale = true` is an error that names the option.

In a graph with loops, or where variational and belief propagation messages meet, the log scale
of a marginal is not the evidence, even where it is defined.

## API

You read a message's log scale with [`getlogscale`](@ref getlogscale(::Message)), and a
marginal's with [`getlogscale`](@ref getlogscale(::Marginal)). Both throw where log scales are not
tracked. An undefined log scale is an
[`UndefinedLogScale`](@extref MessagePassingRulesBase.UndefinedLogScale), and
[`isdefined_logscale`](@extref MessagePassingRulesBase.isdefined_logscale) tells the two apart:

```@example logscale
isdefined_logscale(getlogscale(q)), isdefined_logscale(getlogscale(result))
```

## References

- van Erp, B., Nuijten, W. W. L., van de Laar, T., & de Vries, B. (2023). *Automating Model
  Comparison in Factor Graphs*. Entropy, 25(8), 1138.
  [https://doi.org/10.3390/e25081138](https://doi.org/10.3390/e25081138)
