# What compiling costs every later full GC: the minimum full-collection time after loading the
# packages, and again after the first inference (JIT-compiled code and its metadata now on the
# heap, where the GC marks them; code from package images it does not).
#   julia --project=<env> jit_heap.jl <variant> <model>
const NO_MAIN = true
const PV, PM = ARGS[1], ARGS[2]
empty!(ARGS); append!(ARGS, [PV, PM, "jit", "/dev/null"])
include(joinpath(@__DIR__, "..", "bench_models.jl"))
fullgc() = minimum(@elapsed(GC.gc(true)) for _ in 1:5)
GC.gc(true); t_loaded = fullgc(); live_loaded = Base.gc_live_bytes()
run, iters = setup()
run(max(iters, 1)); run(max(iters, 1))
GC.gc(true); t_run = fullgc(); live_run = Base.gc_live_bytes()
println(join((PV, PM, round(1.0e3 * t_loaded; digits = 2), round(1.0e3 * t_run; digits = 2), round(live_loaded / 1.0e6; digits = 1), round(live_run / 1.0e6; digits = 1)), '\t'))
