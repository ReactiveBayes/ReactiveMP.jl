export FactorBoundFreeEnergy

import Base: tail

struct FactorBoundFreeEnergy end

function score(::Type{T}, ::FactorBoundFreeEnergy, node::AbstractFactorNode, skip_strategy, scheduler) where {T <: CountingReal}
    return score(T, FactorBoundFreeEnergy(), sdtype(node), node, skip_strategy, scheduler)
end

## Deterministic mapping

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Deterministic, node::AbstractFactorNode, skip_strategy, scheduler) where {T <: CountingReal}
    fnstream = let skip_strategy = skip_strategy, scheduler = scheduler
        (interface) -> apply_skip_filter(messagein(interface), skip_strategy) |> schedule_on(scheduler)
    end

    stream = combineLatest(map(fnstream, interfaces(node)), PushNew())

    # TODO: (branch) replace with a function call
    vtag       = Val{inboundclustername(node)}()
    msgs_names = Val{map(name, interfaces(node))}()

    mapping = let fform = functionalform(node), vtag = vtag, meta = metadata(node), msgs_names = msgs_names, node = node
        (messages) -> begin
            # We do not really care about (is_clamped, is_initial) at this stage, so it can be (false, false)
            marginal = Marginal(marginalrule(fform, vtag, msgs_names, messages, nothing, nothing, meta, node), false, false, nothing)
            return convert(T, -score(DifferentialEntropy(), marginal))
        end
    end

    return stream |> map(T, mapping)
end

## Stochastic mapping

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::AbstractFactorNode, skip_strategy, scheduler) where {T <: CountingReal}
    fnstream = let node = node, skip_strategy = skip_strategy, scheduler = scheduler
        (cluster) -> getmarginal!(node, cluster, skip_strategy) |> schedule_on(scheduler)
    end

    stream = combineLatest(map(fnstream, localmarginals(node)), PushNew())

    # TODO: (branch) replace with a function call
    mapping = let fform = functionalform(node), meta = metadata(node), marginal_names = Val{localmarginalnames(node)}()
        (marginals) -> begin
            average_energy   = score(AverageEnergy(), fform, marginal_names, marginals, meta)
            clusters_entropy = mapreduce(marginal -> score(DifferentialEntropy(), marginal), +, marginals)
            return convert(T, average_energy - clusters_entropy)
        end
    end

    return stream |> map(T, mapping)
end
