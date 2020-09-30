export score, AverageEnergy, DifferentialEntropy, BetheFreeEnergy

function score end

struct AverageEnergy end
struct DifferentialEntropy end

struct BetheFreeEnergy end


function score(::BetheFreeEnergy, model::Model, scheduler)
    average_energies = map(filter(isstochastic, getnodes(model))) do node
        marginals = combineLatest(map(cluster -> getmarginal!(node, cluster), clusters(node)), PushEach())
        mapping   = (m) -> score(AverageEnergy(), functionalform(node), m, metadata(node)) - mapreduce(d -> score(DifferentialEntropy(), d), +, m, init = 0.0)
        return marginals |> schedule_on(scheduler) |> map(Float64, mapping) 
    end

    differential_entropies = map(getrandom(model)) do random 
        d       = degree(random)
        mapping = (m) -> (d - 1) * score(DifferentialEntropy(), m)
        return getmarginal(random) |> schedule_on(scheduler) |> map(Float64, mapping)
    end

    energies_sum  = collectLatest(Float64, average_energies, Float64, reduce_with_sum)
    entropies_sum = collectLatest(Float64, differential_entropies, Float64, reduce_with_sum)

    return combineLatest((energies_sum, entropies_sum), PushNew()) |> map(Float64, d -> d[1] + d[2])
end



