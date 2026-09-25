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
`logscale` keyword of [`@define_message_update_rule`](@ref), in one of four forms.

- **A constant**, when the rule's constant does not depend on its inputs:

  ```julia
  @define_message_update_rule(
      node = NormalMeanVariance, target = :out,
      args = (m[:μ]::UnivariateNormalDistributionsFamily, m[:v]::PointMass),
      logscale = 0,
      body = (args) -> NormalMeanVariance(mean(args.m[:μ]), var(args.m[:μ]) + mean(args.m[:v])),
  )
  ```

  Zero is the common case: the convolution of Gaussians integrates to one, and a stochastic
  node's message towards `out` from point-mass inputs is its own normalised density. It is a
  fact about the rule, not a safe default. A Bernoulli node with its output observed at 1 sends
  ``p \mapsto p`` towards `p`, which integrates to ``\tfrac{1}{2}`` over ``[0, 1]``: the rule
  returns `Beta(2, 1)` and declares `logscale = loghalf`. An `Irrational` constant, or an
  integer, takes the precision of whatever it is added to, so a `Float32` model stays `Float32`.

- **A function of the inputs**, over the same slots as the body (`algo`, `ctx`, `args`):

  ```julia
  logscale = (args) -> -length(mean(args.m[:out])) * log(abs(mean(args.m[:A]))),
  ```

- **From the body**, `logscale = from_body`, when the constant shares its work with the result:
  the body returns [`with_logscale`](@ref)`(result, logscale)`, and whoever runs the rule unwraps
  it.

- **Nothing**, when the rule omits the keyword: its message's log scale is an
  [`UndefinedLogScale`](@ref) naming the rule. Every variational rule is such a rule.

A rule that needs the log scales its inputs arrived with declares `reads_logscale = true` and
reads them as `args.logscale.m[:out]`, keyed like `args.m`. `require_logscale` turns an undefined
one into an error that says why; the `Mixture` rules do this.

## What the engine does

Log scales are tracked when nodes are activated with `logscales = true`
(`FactorNodeActivationOptions`; RxInfer's `infer(...; logscales = true)`). Then:

- every message carries the log scale its rule declares, and a message a rule fallback computed
  carries an undefined one;
- observations and constants are point masses, with log scale zero;
- initial messages were computed by no rule: their log scale is undefined;
- a product combines the log scales as above. An undefined side propagates, keeping its reason;
  a pair of distributions with no `compute_logscale` gives an undefined log scale; so does a
  form constraint that changes the product, rather than returning it as it is. A `missing` side leaves the other side's;
- a variable's marginal carries the log scale of the product it was formed from, read with
  [`getlogscale`](@ref).

Nothing errors because a log scale is undefined, except a rule or a user that asks for its value.
Without `logscales = true`, messages carry `nothing`, nothing is computed, and a rule declared
with `reads_logscale = true` is an error that names the option.

In a graph with loops, or where variational and belief-propagation messages meet, the log scale
of a marginal is no longer the evidence, even where it is defined.

## Testing a rule's log scale

A table case states the expected log scale with `ExpectedWithLogScale(value, logscale)`, and the
log scales its inputs arrived with as `logscale = (...)` among its inputs. The table also checks
that a computed log scale follows the inputs' precision. `@verify_message_update_rule` checks the
declared log scale against the node's definition, ``\log \int f \prod_i \mu_i``, by quadrature.

## API

```@docs
ReactiveMP.getlogscale
MessagePassingRulesBase.UndefinedLogScale
MessagePassingRulesBase.UndefinedLogScaleError
MessagePassingRulesBase.require_logscale
MessagePassingRulesBase.isdefined_logscale
MessagePassingRulesBase.with_logscale
MessagePassingRulesBase.WithLogScale
MessagePassingRulesBase.from_body
MessagePassingRulesBase.FromBody
MessagePassingRulesBase.RuleLogScales
```

## References

- van Erp, B., Nuijten, W. W. L., van de Laar, T., & de Vries, B. (2023). *Automating Model Comparison in Factor Graphs*. Entropy, 25(8), 1138. [https://doi.org/10.3390/e25081138](https://doi.org/10.3390/e25081138)
