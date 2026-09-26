# [Log scales](@id lib-logscale)

A message's **log scale** is a scalar carried beside its distribution. A rule returns a
normalised distribution, but what it computes may be an unnormalised function; the log scale is
the constant that relates the two:

```math
\text{message}(x) = \exp(\text{logscale}) \cdot \hat{p}(x),
```

where ``\hat{p}`` is the distribution the rule returns. Where a rule can compute it, it declares
it, and the engine carries it along every message and combines it through every product. Where
it cannot, the log scale is *undefined*, and says why. The same scalar is the one BayesBase and
ExponentialFamily compute for the product of two distributions, `compute_logscale`.

Belief propagation is the common case in which a log scale is known. A belief-propagation
message ``\mu(x) = \int f(x, y, \dots) \prod_i \mu_i(y_i) \,\mathrm{d}y`` is in general not
normalised, and in an acyclic graph the scale factors that such messages carry multiply up to
the model evidence (see below). Nothing restricts the notion to belief propagation, though: any
rule whose result stands for an unnormalised function may declare its constant. A naive
variational message, ``\exp \mathbb{E}_q[\log f]``, has no constant with that meaning, so its
log scale is undefined.

## Why it matters: evidence in message passing

In sum-product message passing on a Forney-style factor graph, a message ``\vec{\mu}_{s_j}(s_j)``
along an edge decomposes as ``\beta_{s_j} \cdot \hat{p}_{s_j}(s_j)``: the normalised
distribution ReactiveMP stores as the message's data, and a positive **scale factor**. A key
result of [van Erp et al. (2023)](https://arxiv.org/abs/2306.05965) is that in an acyclic graph
the product of two colliding messages integrates to exactly the model evidence:

```math
\int \vec{\mu}_{s_j}(s_j)\, \overleftarrow{\mu}_{s_j}(s_j)\, \mathrm{d}s_j = p(y = \hat{y}).
```

So the evidence can be read off locally at *any* edge, from the log scale of a variable's
marginal, in the same run that computes the posteriors. This is what makes **Bayesian model
comparison** — averaging, selection and combination — part of inference: the `Mixture` node's
rules read the log scales of their inputs, and its message towards the switch is a softmax over
the evidence of each component. In a streaming model, the change of a posterior's log scale from
one observation to the next is the one-step predictive likelihood.

## How a product combines them

ReactiveMP tracks the logarithm of the scale factor. When two messages are multiplied, the log
scale of the product is

```math
\log \beta_\text{new} = \log \beta_\text{left} + \log \beta_\text{right} + \log \int \hat{p}_\text{left}(x)\, \hat{p}_\text{right}(x)\, \mathrm{d}x,
```

the last term being `BayesBase.compute_logscale(new, left, right)`. For example, the product of
two normalised Gaussians is not normalised:
``\mathcal{N}(x \mid a, A)\, \mathcal{N}(x \mid b, B) = \mathcal{N}(a \mid b, A + B)\, \mathcal{N}(x \mid c, C)``,
and ``\log \mathcal{N}(a \mid b, A + B)`` is that term. In a Gaussian model, most of the evidence
accumulates here rather than in the rules.

## What a rule declares

The normalised distribution a rule returns does not reveal its log scale: `Beta(2, 1)` looks the
same whether or not a constant was divided out to get it. So a rule states it, with the
`logscale` keyword of
[`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule): a
constant, such as `logscale = 0` for a convolution of Gaussians; a function of the inputs; or
[`from_body`](@extref MessagePassingRulesBase.from_body), the body returning
[`with_logscale`](@extref MessagePassingRulesBase.with_logscale)`(result, logscale)`. A rule that
omits it has an [`UndefinedLogScale`](@extref MessagePassingRulesBase.UndefinedLogScale) naming the
rule: every variational rule is such a rule. A rule that needs the log scales its inputs arrived
with declares `reads_logscale = true` and reads them as `args.logscale.m[:out]`;
[`require_logscale`](@extref MessagePassingRulesBase.require_logscale) turns an undefined one into
an error that says why, as the `Mixture` rules do. How rules declare and test their log scales is
the base package's subject, and the test tools'.

## What the engine does

Log scales are tracked when nodes are activated with `logscales = true`
([`ReactiveMP.FactorNodeActivationOptions`](@ref); RxInfer's `infer(...; logscales = true)`). Then:

- every message carries the log scale its rule declares, and a message a rule fallback computed
  carries an undefined one;
- observations and constants are point masses, with log scale zero;
- initial messages were computed by no rule: their log scale is undefined;
- a product combines the log scales as above. An undefined side propagates, keeping its reason;
  a pair of distributions with no `compute_logscale` gives an undefined log scale; so does a
  form constraint that changes the product, rather than returning it as it is. A `missing` side leaves the other side's;
- a random variable's marginal carries the log scale of the product it was formed from, read
  with [`getlogscale`](@ref getlogscale(::Marginal));
- an initial marginal, set with [`ReactiveMP.set_initial_marginal!`](@ref), and a factor node's
  joint marginal carry none: reading theirs is an error, as where log scales are not tracked.

Nothing errors because a log scale is undefined, except a rule or a user that asks for its value.
Without `logscales = true`, messages carry `nothing`, nothing is computed, and a rule declared
with `reads_logscale = true` is an error that names the option.

In a graph with loops, or where variational and belief-propagation messages meet, the log scale
of a marginal is no longer the evidence, even where it is defined.

## API

A message's log scale is read with [`getlogscale`](@ref getlogscale(::Message)), and a marginal's
with [`getlogscale`](@ref getlogscale(::Marginal)); both throw where log scales are not tracked.
An undefined log scale is an
[`UndefinedLogScale`](@extref MessagePassingRulesBase.UndefinedLogScale), and
[`isdefined_logscale`](@extref MessagePassingRulesBase.isdefined_logscale) tells the two apart.

## References

- van Erp, B., Nuijten, W. W. L., van de Laar, T., & de Vries, B. (2023). *Automating Model Comparison in Factor Graphs*. Entropy, 25(8), 1138. [https://doi.org/10.3390/e25081138](https://doi.org/10.3390/e25081138)
