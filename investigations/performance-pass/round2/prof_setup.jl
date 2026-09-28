# CPU profile of one model's `infer` at its round-1 iteration count (setup-dominated for the BP
# models), on v6 or v7, as inclusive samples per function of the engine's packages.
#   julia --project=<env> prof_setup.jl <variant> <model> <outfile> [reps] [iterations]
# Inclusive: a sample counts once for every distinct function on its stack. The table lists, for
# every function of ReactiveMP, Rocket, RxInfer, GraphPPL, MessagePassingRules* and the C runtime's
# type machinery, its share of all samples; the shares of two variants are then comparable in
# milliseconds through the total time per `infer`.
const NO_MAIN = true
const PV, PM, POUT = ARGS[1], ARGS[2], ARGS[3]
const REPS = parse(Int, get(ARGS, 4, "30"))
const ITERS = get(ARGS, 5, "")
empty!(ARGS); append!(ARGS, [PV, PM, "prof", "/dev/null"])
include(joinpath(@__DIR__, "..", "bench_models.jl"))
using Profile
run, iters = setup()
it = isempty(ITERS) ? max(iters, 1) : parse(Int, ITERS)
run(it); run(it)
GC.gc()
t = minimum(@elapsed(run(it)) for _ in 1:5)
Profile.init(n = 10^8, delay = 0.0001)
Profile.clear()
Profile.@profile for _ in 1:REPS
    run(it)
end
data, lidict = Profile.retrieve()
const PKGS = ("ReactiveMP", "Rocket", "RxInfer", "GraphPPL", "BayesBase", "ExponentialFamily")
# The package a file belongs to: the registry's `packages/<Pkg>/<slug>/src/`, a sibling package
# `lib/<Pkg>/src/`, or the last `<Pkg>.jl/src/` in the path (a scratch path may contain any name).
function package_of(f)
    m = match(r"/packages/([A-Za-z0-9]+)/[^/]+/(src|ext)/", f)
    m !== nothing && return m.captures[1]
    m = match(r"/lib/([A-Za-z0-9]+)/src/", f)
    m !== nothing && return m.captures[1]
    m = match(r".*/([A-Za-z0-9]+)\.jl/src/", f)
    return m === nothing ? nothing : m.captures[1]
end
keep(pkg) = pkg !== nothing && (pkg in PKGS || occursin("MessagePassingRules", pkg))
function analyse()
    incl = Dict{String, Int}(); total = 0; i = 1
    while i <= length(data)
        j = i
        while j <= length(data) && data[j] != 0
            j += 1
        end
        frames = data[i:(j - 1)]; i = j + 1
        isempty(frames) && continue
        total += 1
        seen = Set{String}()
        for ip in frames, sf in get(lidict, ip, Base.StackTraces.StackFrame[])
            f = String(sf.file); fn = String(sf.func)
            key = if sf.from_c
                (startswith(fn, "jl_") || startswith(fn, "ijl_") || occursin("gc", fn)) ? "C:" * fn : continue
            else
                pkg = package_of(f)
                keep(pkg) || continue
                string(pkg, ":", fn)
            end
            key in seen && continue
            push!(seen, key); incl[key] = get(incl, key, 0) + 1
        end
    end
    return incl, total
end
incl, total = analyse()
open(POUT, "w") do io
    println(io, "# variant=$(PV) model=$(PM) iterations=$(it) infer_min_s=$(t) samples=$(total)")
    for (k, v) in sort(collect(incl); by = last, rev = true)
        v < 0.002 * total && break
        println(io, join((k, v, round(100v / total; digits = 2), round(1.0e3 * t * v / total; digits = 3)), '\t'))
    end
end
println("wrote ", POUT, " (", total, " samples, infer ", round(1.0e3t; digits = 1), " ms)")

# FOCUS=<C function>: the innermost Julia frames that call it, by share of the samples it is in.
const FOCUS = get(ENV, "FOCUS", "")
function focus_callers(data, lidict, focus)
    callers = Dict{String, Int}(); n = 0; i = 1
    while i <= length(data)
        j = i
        while j <= length(data) && data[j] != 0
            j += 1
        end
        frames = data[i:(j - 1)]; i = j + 1
        hit = false
        for ip in frames, sf in get(lidict, ip, Base.StackTraces.StackFrame[])
            if !hit
                hit = sf.from_c && String(sf.func) == focus
            elseif !sf.from_c
                k = string(sf.func, " @ ", replace(String(sf.file), r"^.*/(src|lib)/" => s"\1/"), ":", sf.line)
                callers[k] = get(callers, k, 0) + 1; n += 1
                break
            end
        end
    end
    println("callers of ", focus, " (", n, " samples):")
    for (k, v) in first(sort(collect(callers); by = last, rev = true), 25)
        println(lpad(v, 7), "  ", k)
    end
    return
end
isempty(FOCUS) || focus_callers(data, lidict, FOCUS)
