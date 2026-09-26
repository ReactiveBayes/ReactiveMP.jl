"""
    Mixture

A mixture of arbitrary components: `out` is the member of the group `inputs` that the one-hot
`switch` selects,

```math
p(\\mathrm{out} \\mid \\mathrm{switch}, \\mathrm{inputs}) = \\prod_{k=1}^K
δ(\\mathrm{out} - \\mathrm{inputs}_k)^{\\mathrm{switch}_k}.
```

Its interfaces are `out`, `switch` and the group `inputs`, one member per component.

Its rules are belief propagation over the incoming messages, whatever the factorisation, under
its own algorithm, [`MixtureBP`](@ref). Towards `out` the message is a `MixtureDistribution` of
the inputs' messages, towards `switch` a `Categorical` of each component's evidence, and towards
`(:inputs, k)` the message from `out`, scaled by the probability of `k`. The switch's message
must have a `probvec`, a `Categorical` or a `Bernoulli`; the inputs' messages may be of any type
whose product with the message from `out` has a log scale under the algorithm's product
strategy.

The rules read the log scales of their incoming messages (`reads_logscale = true`), so a graph
with a Mixture must track log scales (the engine's activation option `logscales = true`); an
unknown log scale is an error. The node has **no average energy**: the free energy of a model
with one is an error.
"""
struct Mixture end

"""
    MixtureBP(; prod = GenericProd())

[`Mixture`](@ref)'s own algorithm: belief propagation over the incoming messages, **regardless
of the factorisation**. `MixtureBP(GenericProd())` is the node's default, so a model need not
name it.

# Keywords

- `prod`: the product strategy under which the rule towards `switch` multiplies the message
  from `out` with each input's, for the log scale of their product, which is component `k`'s
  evidence. Any BayesBase product strategy. Default `GenericProd()`.

It stands alone, a direct subtype of
[`AbstractAlgorithm`](@extref MessagePassingRulesBase.AbstractAlgorithm). Its dependencies and
rules are declared on `MixtureBP` itself, so they apply for every product strategy.
"""
struct MixtureBP{P} <: AbstractAlgorithm
    prod::P
end

MixtureBP(; prod = GenericProd()) = MixtureBP(prod)

@define_factor_node(node = Mixture, type = Stochastic, interfaces = [:out, :switch, :inputs...], algorithm = MixtureBP)

# Declared on the parametric type, for every product strategy: the node's own `algorithm` keyword
# binds its dependencies and its rules without `algorithm` to its default instance's type.
@define_dependencies(
    node = Mixture, algorithm = MixtureBP,
    dependencies = [
        :out => (m[:switch], m[:inputs...]),
        :switch => (m[:out], m[:inputs...]),
        (:inputs, k) => (m[:out], m[:switch]),
    ],
)

# The log scale an incoming message arrived with, which must be known.
incoming_logscale(logscale) = require_logscale(logscale)

# The log scale of the product of two distributions under the product strategy `strategy`: the
# log of the integral of their product.
function product_logscale(strategy, left, right)
    result = prod(strategy, left, right)
    return BayesBase.compute_logscale(result, left, right)
end
