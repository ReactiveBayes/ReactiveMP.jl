# Line-level CPU profile of one model's `infer`: the innermost line of the engine's packages in
# each sample (its self time plus callees outside them), and the innermost frame of any kind.
#   julia --project=<env> prof_lines.jl <variant> <model> [reps] [iterations]
const NO_MAIN = true
const PV, PM = ARGS[1], ARGS[2]
const REPS = parse(Int, get(ARGS, 3, "10"))
const ITERS = get(ARGS, 4, "")
empty!(ARGS); append!(ARGS, [PV, PM, "prof", "/dev/null"])
include(joinpath(@__DIR__, "..", "bench_models.jl"))
using Profile
run, iters = setup()
it = isempty(ITERS) ? max(iters, 1) : parse(Int, ITERS)
run(it); run(it); GC.gc()
Profile.init(n = 10^8, delay = 0.0001); Profile.clear()
Profile.@profile for _ in 1:REPS
    run(it)
end
data, lidict = Profile.retrieve()
const ENGINE = r"/(ReactiveMP|Rocket|RxInfer|MessagePassingRules[A-Za-z]*|StandardMessagePassingRules|BayesBase|ExponentialFamily)[^/]*/(src|lib)/|/packages/(ReactiveMP|Rocket|RxInfer|BayesBase|ExponentialFamily)/"
function analyse(data, lidict)
    lines = Dict{String, Int}(); anyleaf = Dict{String, Int}(); total = 0; i = 1
    while i <= length(data)
        j = i
        while j <= length(data) && data[j] != 0
            j += 1
        end
        frames = data[i:(j - 1)]; i = j + 1
        isempty(frames) && continue
        sfs0 = get(lidict, frames[1], Base.StackTraces.StackFrame[])
        isempty(sfs0) && continue
        f0 = String(sfs0[1].func)
        f0 == "__psynch_cvwait" && continue   # idle threads
        total += 1
        anyleaf[f0] = get(anyleaf, f0, 0) + 1
        done = false
        for ip in frames, sf in get(lidict, ip, Base.StackTraces.StackFrame[])
            done && break
            sf.from_c && continue
            f = String(sf.file)
            occursin(ENGINE, f) || continue
            k = string(sf.func, " @ ", replace(f, r"^.*/(src|lib)/" => s"\1/"), ":", sf.line)
            lines[k] = get(lines, k, 0) + 1; done = true
        end
    end
    return lines, anyleaf, total
end
lines, anyleaf, total = analyse(data, lidict)
println("busy samples ", total)
println("== innermost engine line")
for (k, v) in first(sort(collect(lines); by = last, rev = true), 45)
    println(lpad(round(100v / total; digits = 1), 6), "%  ", k)
end
println("== innermost frame")
for (k, v) in first(sort(collect(anyleaf); by = last, rev = true), 25)
    println(lpad(round(100v / total; digits = 1), 6), "%  ", k)
end
