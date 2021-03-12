export FactorBoundFreeEnergy

import Base: tail

struct FactorBoundFreeEnergy end

function score(::Type{T}, ::FactorBoundFreeEnergy, node::AbstractFactorNode, scheduler) where { T <: InfCountingReal }
    return score(T, FactorBoundFreeEnergy(), sdtype(node), node, scheduler)
end

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Deterministic, node::AbstractFactorNode, scheduler) where { T <: InfCountingReal }
    stream = combineLatest(map((interface) -> messagein(interface) |> filter(v -> !is_initial(v)) |> schedule_on(scheduler), interfaces(node)), PushNew())

    vtag       = Val{ clustername(tail(interfaces(node))) }
    msgs_names = Val{ map(name, interfaces(node)) }

    mapping = let fform = functionalform(node), vtag = vtag, meta = metadata(node), msgs_names = msgs_names, node = node
        (messages) -> begin
            # We do not really care about (is_clamped, is_initial) at this stage, so it can be (false, false)
            marginal = Marginal(marginalrule(fform, vtag, msgs_names, messages, nothing, nothing, meta, node), false, false)
            return convert(T, -score(DifferentialEntropy(), marginal))
        end
    end

    return stream |> map(T, mapping)
end

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::AbstractFactorNode, scheduler) where { T <: InfCountingReal }
    stream = combineLatest(map((cluster) -> getmarginal!(node, cluster, SkipInitial()) |> schedule_on(scheduler), localmarginals(node)), PushNew())

    mapping = let fform = functionalform(node), meta = metadata(node), marginal_names = Val{ localmarginalnames(node) }
        (marginals) -> begin 
            average_energy   = score(AverageEnergy(), fform, marginal_names, marginals, meta)
            clusters_entropy = mapreduce(marginal -> score(DifferentialEntropy(), marginal), +, marginals)
            return convert(T, average_energy - clusters_entropy)
        end
    end

    return stream |> map(T, mapping)
end