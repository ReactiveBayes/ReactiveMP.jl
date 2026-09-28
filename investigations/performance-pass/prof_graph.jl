# CPU profile of the engine's iid graph loop (no RxInfer), per-iteration part only.
#   julia --project=<env> prof_graph.jl <variant> <n> <outdir> [fe=1]
ENV["BENCH_ONLY"] = "__none__"
const PV, PN, POUT = ARGS[1], parse(Int, ARGS[2]), ARGS[3]
const PFE = get(ARGS, 4, "1") == "1"
empty!(ARGS); push!(ARGS, PV, "prof", "/dev/null")
include(joinpath(@__DIR__, "bench_micro.jl"))
using Profile
function loop(n, it, fe)
    randoms, y, nodes, watched = iid_graph(n)
    foreach(v -> activate!(v, RandomVariableActivationOptions()), randoms)
    foreach(v -> activate!(v, DataVariableActivationOptions()), y)
    foreach(nd -> activate!(nd, FactorNodeActivationOptions()), nodes)
    subs = [subscribe!(get_stream_of_marginals(w), (_) -> nothing) for w in watched]
    fes = fe ? subscribe!(bethe_free_energy(Float64, nodes, [randoms..., y...]), (_) -> nothing) : nothing
    data = randn(n)
    foreach(new_observation!, y, data)
    Profile.clear()
    t = @elapsed Profile.@profile for _ in 1:it
        foreach(new_observation!, y, data)
    end
    return t / it
end
Profile.init(n = 10^8, delay = 0.0002)
loop(PN, 2, PFE)
t = loop(PN, max(4, 2_000_000 ÷ PN), PFE)
println("per-iteration: ", t)
mkpath(POUT)
open(joinpath(POUT, "$(PV)_graph$(PN)_fe$(PFE)_flat.txt"), "w") do io
    Profile.print(IOContext(io, :displaysize => (10000, 400)); format = :flat, sortedby = :count, mincount = 20, C = true)
end
open(joinpath(POUT, "$(PV)_graph$(PN)_fe$(PFE)_tree.txt"), "w") do io
    Profile.print(IOContext(io, :displaysize => (10000, 300)); format = :tree, mincount = 30, C = false, noisefloor = 1.0)
end
