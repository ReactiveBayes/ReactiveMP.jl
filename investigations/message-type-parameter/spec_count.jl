# Compile-side evidence: compile time of each first inference, and the number of method
# specialisations ReactiveMP and Rocket end up with after all five models ran once.
#   julia --project=envA spec_count.jl <outfile>
push!(ARGS, "ssm1", "x", "/dev/null", "/dev/null")  # satisfy bench_models.jl's globals
const OUTF = ARGS[1]
using RxInfer, StableRNGs, LinearAlgebra, Statistics, Serialization, DeltaMessagePassingRules, DiscreteTransitionMessagePassingRules
import Rocket

# reuse the model definitions and setup() without running main()
src = read(joinpath(@__DIR__, "bench_models.jl"), String)
src = replace(src, r"\nmain\(\)\s*$" => "\n")
src = replace(src, r"const MODEL = .*\n" => "", r"const TAG = .*\n" => "", r"const OUT = .*\n" => "", r"const PDIR = .*\n" => "")
include_string(Main, src)

function countspecs(mod)
    total = 0; per = Dict{String, Int}()
    for n in names(mod; all = true)
        isdefined(mod, n) || continue
        f = getfield(mod, n)
        (f isa Function || f isa Type) || continue
        for m in methods(f)
            m.module === mod || continue
            c = count(_ -> true, Base.specializations(m))
            total += c
            per[string(n)] = get(per, string(n), 0) + c
        end
    end
    return total, per
end

Base.cumulative_compile_timing(true)
lines = String[]
for model in ("ssm1", "ssm2", "iid", "nl", "hmm")
    run, iters = setup(model)
    c0 = Base.cumulative_compile_time_ns()[1]
    t = @elapsed run(max(iters, 1))
    c1 = Base.cumulative_compile_time_ns()[1]
    push!(lines, join((VARIANT, model, "first-infer-s", t, "compile-s", (c1 - c0) / 1.0e9), '\t'))
end
for mod in (ReactiveMP, Rocket, RxInfer)
    total, per = countspecs(mod)
    push!(lines, join((VARIANT, string(mod), "specialisations", total), '\t'))
    top = sort(collect(per); by = last, rev = true)[1:15]
    for (k, c) in top
        push!(lines, join((VARIANT, string(mod), "  " * k, c), '\t'))
    end
end
foreach(println, lines)
open(io -> foreach(l -> println(io, l), lines), OUTF, "a")
