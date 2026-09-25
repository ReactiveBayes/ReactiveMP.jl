# `Message` and `Marginal` as `mutable struct`s with `const` fields, or as immutable structs. Times the data-feeding loop of two NormalMeanVariance graphs: an iid model,
# whose one variable of high degree exercises the equality chain, and a chain of BP updates.
# Edit the two struct definitions in `src/message.jl` and `src/marginal.jl` to compare.
#
# Needs ReactiveMP, StandardMessagePassingRules and ExponentialFamily in one environment, e.g.
#   julia -e 'using Pkg; Pkg.activate(temp = true); Pkg.develop([PackageSpec(path = p) for p in
#             (".", "lib/MessagePassingRulesBase", "lib/StandardMessagePassingRules")]);
#             Pkg.add(["ExponentialFamily", "Rocket", "BayesBase"]); include("scripts/benchmark_message_representation.jl")'

using ReactiveMP, Rocket, BayesBase, ExponentialFamily, StandardMessagePassingRules, Statistics
import ReactiveMP: israndom, activate!, FactorNodeActivationOptions, RandomVariableActivationOptions, DataVariableActivationOptions, get_stream_of_marginals

bethe(interfaces) = (Tuple(n for (n, v) in interfaces if israndom(v)), Tuple((n,) for (n, v) in interfaces if !israndom(v))...)
node(fform, interfaces) = factornode(fform, interfaces, filter(!isempty, bethe(interfaces)))

function iid_graph(n)
    x = randomvar(); y = [datavar() for _ in 1:n]; v = constvar(1.0)
    nodes = [node(NormalMeanVariance, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])]
    append!(nodes, [node(NormalMeanVariance, [(:out, y[i]), (:μ, x), (:v, v)]) for i in 1:n])
    return [x], y, nodes, [x]
end

function chain_graph(n)
    x = [randomvar() for _ in 1:n]; y = [datavar() for _ in 1:n]; v = constvar(1.0)
    nodes = Any[node(NormalMeanVariance, [(:out, x[1]), (:μ, constvar(0.0)), (:v, constvar(10.0))])]
    push!(nodes, node(NormalMeanVariance, [(:out, y[1]), (:μ, x[1]), (:v, v)]))
    for i in 2:n
        push!(nodes, node(NormalMeanVariance, [(:out, x[i]), (:μ, x[i - 1]), (:v, v)]))
        push!(nodes, node(NormalMeanVariance, [(:out, y[i]), (:μ, x[i]), (:v, v)]))
    end
    return x, y, nodes, x
end

function run(graph, n, iterations, data)
    randoms, y, nodes, watched = graph(n)
    foreach(v -> activate!(v, RandomVariableActivationOptions()), randoms)
    foreach(v -> activate!(v, DataVariableActivationOptions()), y)
    foreach(nd -> activate!(nd, FactorNodeActivationOptions()), nodes)
    subs = [subscribe!(get_stream_of_marginals(w), (_) -> nothing) for w in watched]
    fe = subscribe!(bethe_free_energy(Float64, nodes, [randoms..., y...]), (_) -> nothing)
    stats = @timed for _ in 1:iterations
        foreach(new_observation!, y, data)
    end
    foreach(unsubscribe!, subs); unsubscribe!(fe)
    return stats
end

function measure(name, graph, n, iterations)
    data = randn(n)
    run(graph, n, iterations, data)
    samples = [run(graph, n, iterations, data) for _ in 1:7]
    t = minimum(s -> s.time, samples); m = median(map(s -> s.time, samples)); b = minimum(s -> s.bytes, samples)
    return println(rpad(name, 28), " min ", round(t * 1.0e3; digits = 2), " ms   median ", round(m * 1.0e3; digits = 2), " ms   ", round(b / 2^20; digits = 2), " MiB")
end

measure("iid n=1000, 10 it", iid_graph, 1000, 10)
measure("iid n=10000, 10 it", iid_graph, 10000, 10)
measure("chain n=300, 10 it", chain_graph, 300, 10)
