# CPU profile of model creation (one stage) for one model and size.
#   julia --project=<env> prof_creation.jl <variant> <model> <n> <stage> <outdir>
const NO_MAIN = true
const PV, PM, PN, PSTAGE, POUT = ARGS[1], ARGS[2], parse(Int, ARGS[3]), ARGS[4], ARGS[5]
empty!(ARGS); append!(ARGS, [PV, "prof", "/dev/null", PM])
include(joinpath(@__DIR__, "bench_creation.jl"))
using Profile
gen, data, cons, meta, init = setup(PM, PN)
create(PSTAGE, gen, data, cons, meta, init); create(PSTAGE, gen, data, cons, meta, init)
Profile.init(n = 10^8, delay = 0.0002); Profile.clear()
reps = parse(Int, get(ENV, "PROF_REPS", "5"))
Profile.@profile for _ in 1:reps
    create(PSTAGE, gen, data, cons, meta, init)
end
mkpath(POUT); tag = "$(PV)_create_$(PM)_$(PN)_$(PSTAGE)"
open(io -> Profile.print(IOContext(io, :displaysize => (10000, 400)); format = :flat, sortedby = :count, mincount = 10, C = false), joinpath(POUT, "$(tag)_flat.txt"), "w")
open(io -> Profile.print(IOContext(io, :displaysize => (10000, 400)); format = :tree, mincount = 30, maxdepth = 70, C = false, noisefloor = 2.0), joinpath(POUT, "$(tag)_tree.txt"), "w")
function bymodule()
    data, lidict = Profile.retrieve()
    bymod = Dict{String, Int}(); total = 0; i = 1
    while i <= length(data)
        j = i
        while j <= length(data) && data[j] != 0
            j += 1
        end
        frames = data[i:(j - 1)]; i = j + 1
        isempty(frames) && continue
        leaf = nothing
        for ip in frames, sf in get(lidict, ip, Base.StackTraces.StackFrame[])
            sf.from_c && continue
            f = String(sf.file)
            leaf = occursin("/lib/", f) && occursin("ReactiveMP.jl", f) ? match(r"lib/([^/]+)/", f)[1] :
                occursin("ReactiveMP.jl/src", f) ? "ReactiveMP" : occursin("Rocket", f) ? "Rocket" : occursin("GraphPPL", f) ? "GraphPPL" :
                occursin("RxInfer", f) ? "RxInfer" : occursin("Dictionaries", f) ? "Dictionaries" : occursin("MetaGraphsNext", f) ? "MetaGraphsNext" :
                occursin("BitSetTuples", f) ? "BitSetTuples" : occursin("Graphs", f) ? "Graphs" : "Base/other"
            break
        end
        total += 1; k = something(leaf, "C/runtime"); bymod[k] = get(bymod, k, 0) + 1
    end
    return bymod, total
end
bymod, total = bymodule()
open(joinpath(POUT, "$(tag)_summary.txt"), "w") do io
    println(io, "samples: ", total)
    for (k, v) in sort(collect(bymod); by = last, rev = true)
        println(io, rpad(k, 30), lpad(v, 8), lpad(round(100v / total; digits = 1), 7), "%")
    end
end
print(read(joinpath(POUT, "$(tag)_summary.txt"), String))
