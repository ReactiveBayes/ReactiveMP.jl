export score, AverageEnergy, DifferentialEntropy, BetheFreeEnergy

function score end

struct AverageEnergy end
struct DifferentialEntropy end

struct BetheFreeEnergy end


function score(::BetheFreeEnergy, model::Model, scheduler)
    average_energies = map(filter(isstochastic, getnodes(model))) do node
        marginals = combineLatest(map(cluster -> getmarginal!(node, cluster), clusters(node)), PushEach())
        mapping   = (m) -> InfCountingReal(score(AverageEnergy(), functionalform(node), m, metadata(node))) - mapreduce(d -> score(DifferentialEntropy(), d), +, m, init = zero(InfCountingReal{Float64}))
        return marginals |> schedule_on(scheduler) |> map(InfCountingReal{Float64}, mapping)
    end

    differential_entropies = map(getrandom(model)) do random 
        d       = degree(random)
        mapping = (m) -> (d - 1) * InfCountingReal(score(DifferentialEntropy(), m))
        return getmarginal(random) |> schedule_on(scheduler) |> map(InfCountingReal{Float64}, mapping)
    end

    energies_sum     = collectLatest(InfCountingReal{Float64}, average_energies, InfCountingReal{Float64}, reduce_with_sum)
    entropies_sum    = collectLatest(InfCountingReal{Float64}, differential_entropies, InfCountingReal{Float64}, reduce_with_sum)
    diracs_entropies = Infinity(length(getdata(model)) + length(getconstant(model)))

    return combineLatest((energies_sum, entropies_sum), PushNew()) |> map(Float64, d -> convert(Float64, d[1] + d[2] + diracs_entropies))
end



