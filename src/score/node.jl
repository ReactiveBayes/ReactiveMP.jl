export FactorBoundFreeEnergy

"""
    FactorBoundFreeEnergy()

Selects a factor node's contribution to the Bethe free energy in [`score`](@ref). A stochastic
node's is its average energy under its local marginals, less the sum of their entropies; a
deterministic node's is minus the entropy of its second cluster: a single input's marginal, or the
joint over its inputs. Under the engine's default dependency scheme its marginal rule computes that
joint from the latest messages on all of its interfaces once none of them is an initial message;
a node whose algorithm declares its dependencies gives the joint its rules read, the cluster's.

The average energy is the one the node's rule package declares for its clusters under the
node's algorithm, found with
[`find_average_energy`](@extref MessagePassingRulesBase.find_average_energy). It runs with the
engine's context, [`ReactiveMP.node_context`](@ref)`(node)`, not with the services of the
activation option `context`, and without the activation's diagnostics.
"""
struct FactorBoundFreeEnergy end

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

# Minus the entropy of the joint over the inputs, the node's second cluster. Under the default
# scheme it is computed by the node's marginal rule from the latest messages on all of its
# interfaces once none of them is an initial message, and again whenever they all have been
# renewed. Not from the cluster's own stream: that one updates as soon as some inputs are renewed,
# so in a graph with loops it can hold a joint computed from an initial message, which no later
# update replaces until every input is renewed, and the free energy of the iterations before
# convergence would not be that of one set of messages. A node whose algorithm declares its
# dependencies has rules that read the cluster, which may be computed by an impure rule, such as
# Delta's under `CVIProjection`: its term is that joint's, from the cluster's stream.
function score(
        ::Type{T},
        ::FactorBoundFreeEnergy,
        ::Deterministic,
        node::AbstractFactorNode,
        algorithm,
        stream_postprocessors,
    ) where {T <: CountingReal}
    joint = last(get_node_local_marginals(getlocalclusters(node)))
    fform = functionalform(node)
    declared = MessagePassingRulesBase.dependencies_spec(fform, algorithm) !== nothing
    # A single input is its variable's own marginal, which is never behind its messages.
    (declared || !isjoint(joint)) && return postprocess_stream_of_scores(
        stream_postprocessors,
        get_stream_of_marginals(joint) |> skip_initial() |> map(T, (marginal) -> convert(T, -score(DifferentialEntropy(), marginal))),
    )
    interfaces = Tuple(getinterfaces(node))
    mapping = marginal_mapping(
        fform,
        MessagePassingRulesBase.ClusterTarget(name(joint)),
        input_names(map(interface -> input_label(node, interface), interfaces)),
        nothing,
        algorithm,
        node,
        EngineDiagnostics(),
        node_context(node),
    )
    messages = combineLatest(map(interface -> get_stream_of_inbound_messages(interface) |> skip_initial(), interfaces), PushNew())
    entropy = let mapping = mapping
        (messages) -> begin
            marginal = has_missing_inputs(messages) || has_missing_statics(node) ? missing : compute_marginal(mapping, messages, nothing)
            return convert(T, -score(DifferentialEntropy(), Marginal(marginal, false, false)))
        end
    end
    stream_of_scores = with_statics(node, messages) |> map(T, entropy)
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
            args = rule_arguments(nothing, nothing, marginals_names, marginals)
            spec = resolve_rule(MessagePassingRulesBase.find_average_energy(fform, algorithm, args))
            MessagePassingRulesBase.check_services(spec, ctx)
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
