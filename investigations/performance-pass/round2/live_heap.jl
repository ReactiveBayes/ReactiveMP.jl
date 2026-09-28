# The cost of the live model to the GC: after one `infer`, with its result (and so the model) kept
# alive, the minimum time of a full collection, and the live bytes GC reports; against the same
# with nothing kept. More live objects per node make every full collection mark more.
#   julia --project=<env> live_heap.jl <variant> <model>
const NO_MAIN = true
const PV, PM = ARGS[1], ARGS[2]
empty!(ARGS); append!(ARGS, [PV, PM, "live", "/dev/null"])
include(joinpath(@__DIR__, "..", "bench_models.jl"))
run, iters = setup()
it = max(iters, 1)
run(it); run(it)
fullgc() = minimum(@elapsed(GC.gc(true)) for _ in 1:5)
GC.gc(true); base_t = fullgc(); base_live = Base.gc_live_bytes()
kept = run(it)
GC.gc(true); kept_t = fullgc(); kept_live = Base.gc_live_bytes()
println(join((PV, PM, round(1.0e3 * base_t; digits = 2), round(1.0e3 * kept_t; digits = 2), round((kept_live - base_live) / 1.0e6; digits = 2)), '\t'))
kept === nothing && println()
