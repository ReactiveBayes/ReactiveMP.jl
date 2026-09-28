# CPU profile of building and activating the engine's chain graph (no RxInfer, no GraphPPL).
#   julia --project=<env> prof_activate.jl <variant> <n> <outdir>
ENV["BENCH_ONLY"] = "__none__"
const PV, PN, POUT = ARGS[1], parse(Int, ARGS[2]), ARGS[3]
empty!(ARGS); push!(ARGS, PV, "prof", "/dev/null")
include(joinpath(@__DIR__, "bench_micro.jl"))
using Profile
function build(n)
    randoms, y, nodes, watched = chain_graph(n)
    t1 = @elapsed foreach(v -> activate!(v, RandomVariableActivationOptions()), randoms)
    t2 = @elapsed foreach(v -> activate!(v, DataVariableActivationOptions()), y)
    t3 = @elapsed foreach(nd -> activate!(nd, FactorNodeActivationOptions()), nodes)
    t4 = @elapsed subs = [subscribe!(get_stream_of_marginals(w), (_) -> nothing) for w in watched]
    t5 = @elapsed fes = subscribe!(bethe_free_energy(Float64, nodes, [randoms..., y...]), (_) -> nothing)
    return (t1, t2, t3, t4, t5)
end
graphonly(n) = @elapsed chain_graph(n)
build(PN); graphonly(PN)
ts = [build(PN) for _ in 1:5]
tg = minimum(graphonly(PN) for _ in 1:5)
println("create variables+factornodes: ", tg)
for (k, name) in enumerate(("activate randoms", "activate data", "activate nodes", "subscribe marginals", "subscribe FE"))
    println(name, ": ", minimum(t[k] for t in ts))
end
Profile.init(n = 10^8, delay = 0.0002)
Profile.clear()
Profile.@profile for _ in 1:parse(Int, get(ENV, "PROF_REPS", "10"))
    build(PN)
end
mkpath(POUT)
open(joinpath(POUT, "$(PV)_activate$(PN)_flat.txt"), "w") do io
    Profile.print(IOContext(io, :displaysize => (10000, 400)); format = :flat, sortedby = :count, mincount = 20, C = true)
end
open(joinpath(POUT, "$(PV)_activate$(PN)_tree.txt"), "w") do io
    Profile.print(IOContext(io, :displaysize => (10000, 300)); format = :tree, mincount = 40, C = false, noisefloor = 1.0)
end
