# Allocations per iteration and in setup by type and by source line, for one model, on v6 or v7:
# deterministic, so unaffected by load on the machine.
#   julia --project=<env> alloc_profile.jl <variant> <model> <outprefix> [I1 I2 rate]
# `Profile.Allocs` samples `run(I1)` and `run(I2)` (after a warm-up) at `rate`; per iteration is
# (count(I2) - count(I1)) / (I2 - I1) / rate for each type and each innermost frame of ReactiveMP,
# Rocket, RxInfer, GraphPPL, BayesBase or ExponentialFamily; setup is count(I1) - I1 * per
# iteration. Writes <outprefix>_types.tsv and <outprefix>_frames.tsv.
const NO_MAIN = true
const _variant, _model, _outprefix = ARGS[1], ARGS[2], ARGS[3]
const I1 = parse(Int, get(ARGS, 4, "1"))
const I2 = parse(Int, get(ARGS, 5, "11"))
const RATE = parse(Float64, get(ARGS, 6, "0.05"))
empty!(ARGS); append!(ARGS, [_variant, _model, "alloc", "/dev/null"])
include(joinpath(@__DIR__, "..", "bench_models.jl"))
using Profile

const PKGS = ("ReactiveMP", "Rocket", "RxInfer", "GraphPPL", "BayesBase", "ExponentialFamily", "MessagePassingRules", "Distributions")

function frame_key(a)
    for fr in a.stacktrace
        f = string(fr.file)
        if any(p -> occursin(p, f), PKGS)
            short = replace(f, r"^.*/(src|lib)/" => s"\1/")
            return string(fr.func, " @ ", short, ":", fr.line)
        end
    end
    return "(outside)"
end

function profile_counts(f)
    Profile.Allocs.clear()
    Random.seed!(1)
    Profile.Allocs.@profile sample_rate = RATE f()
    res = Profile.Allocs.fetch()
    types = Dict{String, Tuple{Float64, Float64}}()
    frames = Dict{String, Tuple{Float64, Float64}}()
    for a in res.allocs
        t = string(a.type)
        c, b = get(types, t, (0.0, 0.0)); types[t] = (c + 1, b + a.size)
        k = frame_key(a)
        c, b = get(frames, k, (0.0, 0.0)); frames[k] = (c + 1, b + a.size)
    end
    return types, frames
end

using Random
run, iters = setup()
iters == 0 && error("$(_model) is not iterative")
run(I1); run(I2)
GC.gc()
t1, f1 = profile_counts(() -> run(I1))
t2, f2 = profile_counts(() -> run(I2))

function write_table(path, a, b)
    ks = union(keys(a), keys(b))
    rows = map(collect(ks)) do k
        c1, b1 = get(a, k, (0.0, 0.0)); c2, b2 = get(b, k, (0.0, 0.0))
        pc = (c2 - c1) / (I2 - I1) / RATE; pb = (b2 - b1) / (I2 - I1) / RATE
        sc = c1 / RATE - I1 * pc; sb = b1 / RATE - I1 * pb
        (k, pc, pb, sc, sb)
    end
    sort!(rows; by = r -> -r[3])
    open(path, "w") do io
        println(io, join(("key", "per_iter_count", "per_iter_bytes", "setup_count", "setup_bytes"), '\t'))
        for r in rows
            println(io, join((r[1], round(r[2]; digits = 1), round(r[3]; digits = 0), round(r[4]; digits = 0), round(r[5]; digits = 0)), '\t'))
        end
    end
    return rows
end
rt = write_table(_outprefix * "_types.tsv", t1, t2)
write_table(_outprefix * "_frames.tsv", f1, f2)
tot(rows, i) = sum(r -> r[i], rows)
println("per iteration: ", round(tot(rt, 2)), " allocations, ", round(tot(rt, 3) / 1.0e6; digits = 3), " MB; setup: ", round(tot(rt, 4)), " allocations, ", round(tot(rt, 5) / 1.0e6; digits = 3), " MB")
