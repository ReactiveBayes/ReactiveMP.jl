# Exact bytes and allocation counts per iteration and in setup (`@timed`, deterministic), from
# T(I) and T(2I). julia --project=<env> bytes_per_iter.jl <variant> <model> [I]
const NO_MAIN = true
const PV, PM = ARGS[1], ARGS[2]
const II = parse(Int, get(ARGS, 3, "20"))
empty!(ARGS); append!(ARGS, [PV, PM, "bytes", "/dev/null"])
include(joinpath(@__DIR__, "..", "bench_models.jl"))
run, iters = setup()
run(II); run(2II)
a = @timed run(II); b = @timed run(2II)
ca, cb = Base.gc_alloc_count(a.gcstats), Base.gc_alloc_count(b.gcstats)
println(join((PV, PM, round((b.bytes - a.bytes) / II / 1.0e3; digits = 1), round((cb - ca) / II; digits = 0), round((a.bytes - II * (b.bytes - a.bytes) / II) / 1.0e6; digits = 2), round(ca - II * (cb - ca) / II; digits = 0)), '\t'))
