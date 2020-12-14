export FactorBoundFreeEnergy

import Base: tail

struct FactorBoundFreeEnergy end

function score(::Type{T}, ::FactorBoundFreeEnergy, node::AbstractFactorNode, scheduler) where T
    return score(T, FactorBoundFreeEnergy(), sdtype(node), node, scheduler)
end

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Deterministic, node::AbstractFactorNode, scheduler) where T
    stream = combineLatest(map((interface) -> messagein(interface), interfaces(node)), PushNew())

    vtag       = Val{ clustername(tail(interfaces(node))) }
    msgs_names = Val{ map(name, interfaces(node)) }

    mapping = let fform = functionalform(node), vtag = vtag, meta = metadata(node), msgs_names = msgs_names, node = node
        (messages) -> begin
            marginal = as_marginal(marginalrule(fform, vtag, msgs_names, messages, nothing, nothing, meta, node))
            return convert(InfCountingReal{T}, -score(DifferentialEntropy(), marginal))
        end
    end

    return stream |> schedule_on(scheduler) |> map(InfCountingReal{T}, mapping)
end

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::AbstractFactorNode, scheduler) where T
    stream = combineLatest(map((cluster) -> getmarginal!(node, cluster), localmarginals(node)), PushEach())

    mapping = let fform = functionalform(node), meta = metadata(node), marginal_names = Val{ localmarginalnames(node) }
        (marginals) -> begin 
            average_energy   = convert(InfCountingReal{T}, score(AverageEnergy(), fform, marginal_names, marginals, meta))
            clusters_entropy = mapreduce(marginal -> score(DifferentialEntropy(), marginal), +, marginals)
            return convert(InfCountingReal{T}, average_energy - clusters_entropy)
        end
    end

    return stream |> schedule_on(scheduler) |> map(InfCountingReal{T}, mapping)
end