"""
    Mixture

A mixture of arbitrary components: `out` is the input `switch` selects, one member of the
group `inputs` per component. Its rules are belief propagation over the incoming messages,
whatever the factorisation, under its own algorithm, [`MixtureBP`](@ref).

The rules read the log scales of their incoming messages (`reads_logscale = true`), so a graph
with a Mixture tracks log scales. The node has no average energy: a free energy of a model with
one is an error.
"""
struct Mixture end

"""
    MixtureBP(; prod = GenericProd())

[`Mixture`](@ref)'s own algorithm: belief propagation over the incoming messages, **regardless
of the factorisation**. `prod` is the product strategy the rule towards `switch` multiplies the
message from `out` with each input's under, for the log scale of their product.
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
