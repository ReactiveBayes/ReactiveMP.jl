export FactorBoundFreeEnergy

import Base: tail

struct FactorBoundFreeEnergy end

function score(::Type{T}, ::FactorBoundFreeEnergy, node::AbstractFactorNode, scheduler) where { T <: InfCountingReal }
    return score(T, FactorBoundFreeEnergy(), sdtype(node), node, scheduler)
end

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Deterministic, node::AbstractFactorNode, scheduler) where { T <: InfCountingReal }
    stream = combineLatest(map((interface) -> messagein(interface) |> schedule_on(scheduler), interfaces(node)), PushNew())

    vtag       = Val{ clustername(tail(interfaces(node))) }
    msgs_names = Val{ map(name, interfaces(node)) }

    mapping = let fform = functionalform(node), vtag = vtag, meta = metadata(node), msgs_names = msgs_names, node = node
        (messages) -> begin

            # Marginal is clamped if all of the inputs are clamped
            is_marginal_clamped = __check_all(is_clamped, messages)

            # Marginal is initial if it is not clamped and all of the inputs are either clamped or initial
            is_marginal_initial = !is_marginal_clamped && (__check_all(m -> is_clamped(m) || is_initial(m), messages))

            marginal = Marginal(marginalrule(fform, vtag, msgs_names, messages, nothing, nothing, meta, node), is_marginal_clamped, is_marginal_initial)
            
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