"""
    Mixture

A mixture of arbitrary components: `out` is the input `switch` selects, one member of the
group `inputs` per component. Its rules are belief propagation over the incoming messages,
whatever the factorisation, under its own algorithm, [`MixtureBP`](@ref).

The rules need the log scales of their incoming messages, so a graph with a Mixture needs
log-scale annotations on its messages. The node has no average energy: a free energy of a model
with one is an error.
"""
struct Mixture end

"""
    MixtureBP()

[`Mixture`](@ref)'s own algorithm: belief propagation over the incoming messages, **regardless
of the factorisation**.
"""
struct MixtureBP <: AbstractAlgorithm end

@define_factor_node(
    node = Mixture,
    type = Stochastic,
    interfaces = [:out, :switch, :inputs...],
    algorithm = MixtureBP,
    dependencies = [
        :out => (m[:switch], m[:inputs...]),
        :switch => (m[:out], m[:inputs...]),
        (:inputs, k) => (m[:out], m[:switch]),
    ],
)

const MIXTURE_NEEDS_LOGSCALES = "The Mixture node needs the log scale of every incoming message. Enable log-scale annotations (`LogScaleAnnotations`) for the model."

# The log scale an incoming message arrived with, read from its annotations.
incoming_logscale(annotations) =
    hasannotation(annotations, :logscale) ? getannotation(annotations, :logscale) : error(MIXTURE_NEEDS_LOGSCALES)
