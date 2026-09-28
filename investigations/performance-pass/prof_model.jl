# CPU profile of one model's steady-state `infer`, grouped by module and by function.
#   julia --project=<env> prof_model.jl <variant> <model> <outdir> [iterations-multiplier]
const NO_MAIN = true
push!(ARGS, "prof", "/dev/null")   # bench_models.jl reads ARGS[3], ARGS[4]
const PVARIANT, PMODEL, POUT = ARGS[1], ARGS[2], ARGS[3]
ARGS[1], ARGS[2] = PVARIANT, PMODEL
ARGS[3], ARGS[4] = "prof", "/dev/null"
include(joinpath(@__DIR__, "bench_models.jl"))
using Profile
run, iters = setup()
it = max(iters, 1) * parse(Int, get(ENV, "PROF_ITERS_MULT", "1"))
run(it); run(it)
Profile.init(n = 10^8, delay = 0.0002)
Profile.clear()
reps = parse(Int, get(ENV, "PROF_REPS", "20"))
Profile.@profile for _ in 1:reps
    run(it)
end
mkpath(POUT)
tag = "$(PVARIANT)_$(PMODEL)"
open(joinpath(POUT, "$(tag)_flat.txt"), "w") do io
    Profile.print(IOContext(io, :displaysize => (10000, 400)); format = :flat, sortedby = :count, mincount = 20, C = false)
end
open(joinpath(POUT, "$(tag)_tree.txt"), "w") do io
    Profile.print(IOContext(io, :displaysize => (10000, 400)); format = :tree, mincount = 50, maxdepth = 60, C = false, noisefloor = 2.0)
end
# self time by module of the leaf frame and inclusive time of marker functions
data, lidict = Profile.retrieve()
function analyse()
    bymod = Dict{String, Int}(); total = 0
    incl = Dict{String, Int}()
    markers = ["compute_product_of_messages", "MessageMapping", "MarginalMapping", "execute_rule", "materialize!", "create_model", "activate!", "_compute_marginal_from_messages", "new_observation!", "score", "combineLatest", "collectLatest", "gc", "jl_gc", "Message", "AfterMessageRuleCallEvent", "BeforeMessageRuleCallEvent", "generate_span_id", "constrain_form", "AnnotationDict", "jl_apply_generic", "ijl_apply_generic", "_compute_sparams", "jl_f__compute_sparams", "prod"]
    i = 1
    while i <= length(data)
        # a backtrace ends with 0 (and on 1.13 with some metadata words); walk frames until 0
        j = i
        while j <= length(data) && data[j] != 0
            j += 1
        end
        frames = data[i:(j - 1)]
        i = j + 1
        isempty(frames) && continue
        seen = Set{String}()
        leafmod = nothing
        for (k, ip) in enumerate(frames)
            for sf in get(lidict, ip, Base.StackTraces.StackFrame[])
                fname = String(sf.func)
                file = String(sf.file)
                if leafmod === nothing && !sf.from_c
                    leafmod = occursin("ReactiveMP.jl/lib/", file) ? match(r"lib/([^/]+)/", file)[1] :
                        occursin("ReactiveMP.jl/src", file) || occursin("ReactiveMP/", file) ? "ReactiveMP" :
                        occursin("Rocket", file) ? "Rocket" : occursin("GraphPPL", file) ? "GraphPPL" :
                        occursin("RxInfer", file) ? "RxInfer" : occursin("ExponentialFamily", file) ? "ExponentialFamily" :
                        occursin("BayesBase", file) ? "BayesBase" : occursin("Distributions", file) ? "Distributions" :
                        occursin("Dictionaries", file) ? "Dictionaries" : occursin("MetaGraphsNext", file) ? "MetaGraphsNext" :
                        startswith(file, "./") || occursin("share/julia", file) || occursin("base/", file) ? "Base" : file
                end
                for m in markers
                    if (fname == m || occursin(m, fname)) && !(m in seen)
                        push!(seen, m); incl[m] = get(incl, m, 0) + 1
                    end
                end
            end
        end
        total += 1
        k = something(leafmod, "C/runtime")
        bymod[k] = get(bymod, k, 0) + 1
    end
    return bymod, incl, total
end
bymod, incl, total = analyse()
open(joinpath(POUT, "$(tag)_summary.txt"), "w") do io
    println(io, "samples: ", total, "  (", reps, " x infer, iterations = ", it, ")")
    println(io, "\n# self samples by module of the innermost Julia frame")
    for (k, v) in sort(collect(bymod); by = last, rev = true)
        println(io, rpad(k, 40), lpad(v, 8), lpad(round(100v / total; digits = 1), 7), "%")
    end
    println(io, "\n# inclusive samples of marker functions (substring match)")
    for (k, v) in sort(collect(incl); by = last, rev = true)
        println(io, rpad(k, 40), lpad(v, 8), lpad(round(100v / total; digits = 1), 7), "%")
    end
end
print(read(joinpath(POUT, "$(tag)_summary.txt"), String))
