# CPU profile of activating N nodes of one shape (bench_setup_micro.jl's shapes), inclusive per
# function, and the innermost engine lines of the samples in the C runtime's type machinery.
#   julia --project=<variant>/env prof_activate2.jl <shape index> [reps]
const NO_RUN = true
include(joinpath(@__DIR__, "bench_setup_micro.jl"))
using Profile
const SHAPE = SHAPES[parse(Int, ARGS[1])]
const REPS = parse(Int, get(ARGS, 2, "20"))
let (name, fform, keys, factorisation, kinds) = SHAPE
    create = inputs -> [factornode(fform, i, factorisation) for i in inputs]
    options = FactorNodeActivationOptions()
    function prepared()
        inputs = shape_inputs(keys, kinds); nodes = create(inputs); product = MessageProductContext()
        for i in inputs, (_, v) in i
            israndom(v) && activate!(v, RandomVariableActivationOptions(nothing, product, product))
            isdata(v) && activate!(v, DataVariableActivationOptions(true, false, nothing, nothing))
        end
        return nodes
    end
    batches = [prepared() for _ in 1:(REPS + 1)]
    foreach(node -> activate!(node, options), batches[1])
    Profile.init(n = 10^8, delay = 0.00005); Profile.clear()
    Profile.@profile for b in batches[2:end]
        foreach(node -> activate!(node, options), b)
    end
end
data, lidict = Profile.retrieve()
incl = Dict{String, Int}(); leafs = Dict{String, Int}(); total = 0; i = 1
while i <= length(data)
    global i, total
    j = i
    while j <= length(data) && data[j] != 0
        j += 1
    end
    frames = data[i:(j - 1)]; i = j + 1
    isempty(frames) && continue
    total += 1; seen = Set{String}(); leaf = true
    for ip in frames, sf in get(lidict, ip, Base.StackTraces.StackFrame[])
        f = String(sf.file)
        sf.from_c && continue
        occursin(r"(ReactiveMP|Rocket|MessagePassingRules)[^/]*/(src|lib)", f) || continue
        k = string(sf.func, " @ ", replace(f, r"^.*/(src|lib)/" => s"\1/"), ":", sf.line)
        if leaf
            leafs[k] = get(leafs, k, 0) + 1; leaf = false
        end
        kf = string(sf.func, " @ ", replace(f, r"^.*/(src|lib)/" => s"\1/"))
        kf in seen && continue
        push!(seen, kf); incl[kf] = get(incl, kf, 0) + 1
    end
end
println("samples ", total)
println("== inclusive by function")
for (k, v) in first(sort(collect(incl); by = last, rev = true), 45)
    println(lpad(round(100v / total; digits = 1), 6), "%  ", k)
end
println("== innermost engine line (self + callees outside the engine)")
for (k, v) in first(sort(collect(leafs); by = last, rev = true), 40)
    println(lpad(round(100v / total; digits = 1), 6), "%  ", k)
end
# C-runtime self time: the innermost frame of every sample, C included
cself = Dict{String, Int}()
let i = 1
    while i <= length(data)
        j = i
        while j <= length(data) && data[j] != 0
            j += 1
        end
        frames = data[i:(j - 1)]; i = j + 1
        isempty(frames) && continue
        sfs = get(lidict, frames[1], Base.StackTraces.StackFrame[])
        isempty(sfs) && continue
        k = String(sfs[1].func); cself[k] = get(cself, k, 0) + 1
    end
end
println("== innermost frame (any)")
for (k, v) in first(sort(collect(cself); by = last, rev = true), 25)
    println(lpad(round(100v / total; digits = 1), 6), "%  ", k)
end
