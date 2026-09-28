# Where time to first inference goes: SnoopCompile's inference timing, by module and by root.
#   julia --project=<env> snoop_ttfx.jl <variant> <model> <outfile>
using SnoopCompileCore
const SV, SM, SOUT = ARGS[1], ARGS[2], ARGS[3]
const NO_MAIN = true
empty!(ARGS); push!(ARGS, SV, SM, "snoop", "/dev/null")
include(joinpath(@__DIR__, "bench_models.jl"))
run, iters = setup()
tinf = @snoop_inference run(max(iters, 1))
using SnoopCompile
open(SOUT, "a") do io
    println(io, "######## ", SV, " ", SM)
    println(io, tinf)
    # exclusive inference time by module of the method inferred
    ftimes = flatten(tinf)
    bymod = Dict{String, Float64}()
    for f in ftimes
        mi = SnoopCompile.MethodInstance(f)
        m = mi.def isa Method ? string(mi.def.module) : "toplevel"
        bymod[m] = get(bymod, m, 0.0) + exclusive(f)
    end
    total = sum(values(bymod))
    println(io, "exclusive inference time by module (total ", round(total; digits = 2), " s):")
    for (k, v) in first(sort(collect(bymod); by = last, rev = true), 15)
        println(io, "  ", rpad(k, 40), round(v; digits = 3), " s  ", round(100v / total; digits = 1), "%")
    end
    # the costliest inference triggers (roots)
    println(io, "top inference roots by inclusive time:")
    for f in last(sort(flatten(tinf; tmin = 0.05); by = inclusive), 0)
    end
    it = inference_triggers(tinf)
    println(io, "inference triggers: ", length(it))
    mtrigs = accumulate_by_source(Method, it)
    for m in last(mtrigs, 15)
        println(io, "  ", first(sprint(show, m), 220))
    end
end
