export FactorBoundFreeEnergy

import Base: tail

struct FactorBoundFreeEnergy end

function score(
    ::Type{T},
    ::FactorBoundFreeEnergy,
    node::AbstractFactorNode,
    meta,
    scheduler,
) where {T <: CountingReal}
    return score(
        T,
        FactorBoundFreeEnergy(),
        sdtype(node),
        node,
        collect_meta(functionalform(node), meta),
        scheduler,
    )
end

## Deterministic mapping

function score(
    ::Type{T},
    ::FactorBoundFreeEnergy,
    ::Deterministic,
    node::AbstractFactorNode,
    meta,
    scheduler,
) where {T <: CountingReal}
    fnstream = let scheduler = scheduler
        (interface) ->
            get_stream_of_inbound_messages(interface) |>
            skip_initial() |>
            schedule_on(scheduler)
    end

    tinterfaces = Tuple(getinterfaces(node))
    stream = combineLatest(map(fnstream, tinterfaces), PushNew())

    vtag       = Val{clustername(getinboundinterfaces(node))}()
    msgs_names = Val{map(name, tinterfaces)}()

    mapping =
        let fform = functionalform(node),
            vtag = vtag,
            msgs_names = msgs_names,
            node = node

            (messages) -> begin
                # We do not really care about (is_clamped, is_initial) at this stage, so it can be (false, false)
                marginal = Marginal(
                    marginalrule(
                        fform,
                        vtag,
                        msgs_names,
                        messages,
                        nothing,
                        nothing,
                        meta,
                        node,
                    ),
                    false,
                    false,
                )
                return convert(T, -score(DifferentialEntropy(), marginal))
            end
        end

    return stream |> map(T, mapping)
end

## Stochastic mapping

function score(
    ::Type{T},
    ::FactorBoundFreeEnergy,
    ::Stochastic,
    node::AbstractFactorNode,
    meta,
    scheduler,
) where {T <: CountingReal}
    fnstream = let scheduler = scheduler
        (localmarginal) ->
            get_stream_of_marginals(localmarginal) |>
            skip_initial() |>
            schedule_on(scheduler)
    end

    localmarginals = get_node_local_marginals(getlocalclusters(node))
    stream = combineLatest(map(fnstream, localmarginals), PushNew())

    mapping =
        let fform = functionalform(node),
            marginal_names = Val{Tuple(map(name, localmarginals))}()

            (marginals) -> begin
                average_energy   = score(AverageEnergy(), fform, marginal_names, marginals, meta)
                clusters_entropy = mapreduce(marginal -> score(DifferentialEntropy(), marginal), +, marginals)
                return convert(T, average_energy - clusters_entropy)
            end
        end

    return stream |> map(T, mapping)
end
