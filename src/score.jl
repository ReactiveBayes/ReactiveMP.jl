export score, AverageEnergy, DifferentialEntropy, BetheFreeEnergy

function score end

struct AverageEnergy end
struct DifferentialEntropy end

struct BetheFreeEnergy end

# TODO: Messages around nodes, not marginals
# TODO: Check if we use clusters instead of marginals
# TODO __score_getmarginal wont work for clusters?
function score(::BetheFreeEnergy, model::Model, scheduler)
    average_energies = map(filter(isstochastic, getnodes(model))) do node
        marginals = combineLatest(map(cluster -> getmarginal!(node, cluster), clusters(node)), PushEach())
        # return marginals |> schedule_on(scheduler) |> map(Float64, (m) -> score(AverageEnergy(), functionalform(node), m) - mapreduce(d -> score(DifferentialEntropy(), d), +, m, init = 0.0)) 
        # below without H[q_a]
        return marginals |> schedule_on(scheduler) |> map(Float64, (m) -> score(AverageEnergy(), functionalform(node), m)) 
    end

    differential_entropies = map(getrandom(model)) do random 
        d = degree(random)
        return getmarginal(random) |> schedule_on(scheduler) |> map(Float64, (m) -> (d - 1) * score(DifferentialEntropy(), m))
    end

    energies_sum  = collectLatest(Float64, average_energies) |> map(Float64, energies -> reduce(+, energies))
    entropies_sum = collectLatest(Float64, differential_entropies) |> map(Float64, entropies -> reduce(+, entropies))

    return combineLatest((energies_sum, entropies_sum), PushNew()) |> map(Float64, d -> d[1] - d[2])
end



