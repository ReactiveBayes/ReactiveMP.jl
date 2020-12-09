export FactorBoundFreeEnergy

struct FactorBoundFreeEnergy end

function score(::Type{T}, ::FactorBoundFreeEnergy, node::AbstractFactorNode, scheduler) where T
    return score(T, FactorBoundFreeEnergy(), sdtype(node), node, scheduler)
end

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Deterministic, node::AbstractFactorNode, scheduler) where T
    error("Factor-Bound Free Energy is not implement for deterministic nodes")
end

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::AbstractFactorNode, scheduler) where T
    stream = combineLatest(map(cluster -> getmarginal!(node, cluster), localmarginals(node)), PushEach())

    mapping = let fform = functionalform(node), meta = metadata(node), marginal_names = Val{ localmarginalnames(node) }
        (marginals) -> begin 
            average_energy   = convert(InfCountingReal{T}, score(AverageEnergy(), fform, marginal_names, marginals, meta))
            clusters_entropy = mapreduce(marginal -> score(DifferentialEntropy(), marginal), +, marginals)
            return convert(InfCountingReal{T}, average_energy - clusters_entropy)
        end
    end

    return stream |> schedule_on(scheduler) |> map(InfCountingReal{T}, mapping)
end