export FactorBoundFreeEnergy

import Base: tail

"""
    FactorBoundFreeEnergy()

Selects a factor node's contribution to the Bethe free energy in [`score`](@ref): its average
energy over each of its clusters, less the entropies of the clusters it scores.
"""
struct FactorBoundFreeEnergy end

"""
    score(::Type{T}, ::FactorBoundFreeEnergy, node, algorithm, stream_postprocessors)

The stream of a factor node's contribution to the Bethe free energy, one value of type `T`
per update. `algorithm` is the one the node runs under; `nothing` means its default.

A stochastic node contributes its average energy minus the entropies of its local marginals;
a deterministic node minus the entropy of the joint over its inputs, its second cluster.
"""
function score(
        ::Type{T},
        ::FactorBoundFreeEnergy,
        node::AbstractFactorNode,
        algorithm,
        stream_postprocessors,
    ) where {T <: CountingReal}
    fform = functionalform(node)
    return score(
        T,
        FactorBoundFreeEnergy(),
        sdtype(node),
        node,
        something(algorithm, default_algorithm(fform)),
        stream_postprocessors,
    )
end

## Deterministic mapping

# Minus the entropy of the joint over the inputs, the node's second cluster, each time it
# updates.
function score(
        ::Type{T},
        ::FactorBoundFreeEnergy,
        ::Deterministic,
        node::AbstractFactorNode,
        algorithm,
        stream_postprocessors,
    ) where {T <: CountingReal}
    joint = last(get_node_local_marginals(getlocalclusters(node)))
    stream_of_scores = get_stream_of_marginals(joint) |> skip_initial() |> map(T, (marginal) -> convert(T, -score(DifferentialEntropy(), marginal)))
    return postprocess_stream_of_scores(stream_postprocessors, stream_of_scores)
end

## Stochastic mapping

function score(
        ::Type{T},
        ::FactorBoundFreeEnergy,
        ::Stochastic,
        node::AbstractFactorNode,
        algorithm,
        stream_postprocessors,
    ) where {T <: CountingReal}
    fnstream = (localmarginal) -> get_stream_of_marginals(localmarginal) |> skip_initial()

    clusters = getlocalclusters(node)
    localmarginals = get_node_local_marginals(clusters)
    stream = combineLatest(map(fnstream, localmarginals), PushNew())

    mapping = let fform = functionalform(node),
            marginals_names = input_names(map(i -> cluster_label(node, clusters, i), Tuple(eachindex(localmarginals)))),
            ctx = node_context(node),
            algorithm = algorithm

        (marginals) -> begin
            args = RuleArgs(rule_messages(getdata, nothing, nothing), rule_marginals(getdata, marginals_names, marginals))
            spec = resolve_rule(MessagePassingRulesBase.find_average_energy(fform, algorithm, args))
            ann = rule_annotations(nothing, nothing, marginals_names, marginals, MessagePassingRulesBase.NoAnnotations())
            average_energy = MessagePassingRulesBase.execute_rule(
                spec, nothing, MessagePassingRulesBase.rule_algorithm(spec, algorithm), ctx, args, ann, nothing
            )
            clusters_entropy = mapreduce(marginal -> score(DifferentialEntropy(), marginal), +, marginals)
            return convert(T, average_energy - clusters_entropy)
        end
    end

    stream_of_scores = stream |> map(T, mapping)
    stream_of_scores = postprocess_stream_of_scores(stream_postprocessors, stream_of_scores)

    return stream_of_scores
end
