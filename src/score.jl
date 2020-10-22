export score, AverageEnergy, DifferentialEntropy, BetheFreeEnergy

function score end

struct AverageEnergy end
struct DifferentialEntropy end

struct BetheFreeEnergy end

function score(::BetheFreeEnergy, model, scheduler)
    return score(Float64, BetheFreeEnergy(), model, scheduler)
end

function score(::Type{T}, ::BetheFreeEnergy, model::Model, scheduler) where T
    average_energies = map(filter(isstochastic, getnodes(model))) do node
        marginals = combineLatest(map(cluster -> getmarginal!(node, cluster), clusters(node)), PushEach())
        mapping   = (m) -> InfCountingReal(score(AverageEnergy(), functionalform(node), m, metadata(node))) - mapreduce(d -> score(DifferentialEntropy(), d), +, m)
        return marginals |> schedule_on(scheduler) |> map(InfCountingReal{T}, mapping)
    end

    differential_entropies = map(getrandom(model)) do random 
        d       = degree(random)
        mapping = (m) -> (d - 1) * InfCountingReal(score(DifferentialEntropy(), m))
        return getmarginal(random) |> schedule_on(scheduler) |> map(InfCountingReal{T}, mapping)
    end

    energies_sum     = collectLatest(InfCountingReal{T}, average_energies, InfCountingReal{T}, reduce_with_sum) 
    entropies_sum    = collectLatest(InfCountingReal{T}, differential_entropies, InfCountingReal{T}, reduce_with_sum) 
    diracs_entropies = Infinity(length(getdata(model)) + length(getconstant(model)))

    return combineLatest((energies_sum, entropies_sum), PushNew()) |> map(T, d -> convert(T, d[1] + d[2] + diracs_entropies))
end



