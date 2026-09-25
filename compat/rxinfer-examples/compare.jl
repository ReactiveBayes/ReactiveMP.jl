# Compares `results/<name>_v6.jls` with `results/<name>_v7.jls`, entry by entry. Plain Julia
# (only Serialization and Printf), runnable in any environment:
#
#     julia compat/rxinfer-examples/compare.jl                 # every example with both files
#     julia compat/rxinfer-examples/compare.jl kalman hmm      # some of them
#     ATOL=1e-6 julia compat/rxinfer-examples/compare.jl       # another tolerance
#
# An entry agrees when every value is within `ATOL` (absolute, default 1e-8) of v6's. The report
# gives, per group of entries (the name before the first `[`), the number of entries, the largest
# absolute and relative difference, and the free-energy values of both sides.

using Serialization, Printf

const RESULTS_DIR = joinpath(@__DIR__, "results")
const ATOL = parse(Float64, get(ENV, "ATOL", "1e-8"))

group(name) = first(split(name, '['))

function compare_example(example)
    v6 = deserialize(joinpath(RESULTS_DIR, "$(example)_v6.jls"))
    v7 = deserialize(joinpath(RESULTS_DIR, "$(example)_v7.jls"))
    println("== $example (atol = $ATOL)")
    only6 = setdiff(keys(v6.values), keys(v7.values))
    only7 = setdiff(keys(v7.values), keys(v6.values))
    isempty(only6) || println("  only in v6: ", join(sort(collect(only6)), ", "))
    isempty(only7) || println("  only in v7: ", join(sort(collect(only7)), ", "))
    ok = isempty(only6) && isempty(only7)

    stats = Dict{String, Any}()
    for name in intersect(keys(v6.values), keys(v7.values))
        a, b = v6.values[name], v7.values[name]
        g = group(name)
        s = get!(stats, g, (n = 0, abs = 0.0, rel = 0.0, bad = 0, worst = "", shape = true, types = Set{Tuple{String, String}}()))
        if length(a) != length(b)
            stats[g] = merge(s, (n = s.n + 1, bad = s.bad + 1, shape = false, worst = name))
            continue
        end
        # A NaN on both sides (a missing observation, recorded as NaN) agrees; on one side it does not.
        Δ(x, y) = (isnan(x) && isnan(y)) ? 0.0 : (isnan(x) || isnan(y)) ? Inf : abs(x - y)
        d = maximum((Δ(x, y) for (x, y) in zip(a, b)); init = 0.0)
        r = maximum((Δ(x, y) / (isnan(x) ? 1.0 : max(abs(x), eps())) for (x, y) in zip(a, b)); init = 0.0)
        push!(s.types, (v6.types[name], v7.types[name]))
        stats[g] = merge(
            s, (
                n = s.n + 1, abs = max(s.abs, d), rel = max(s.rel, r), bad = s.bad + (d > ATOL),
                worst = d > s.abs ? name : s.worst,
            )
        )
    end

    for g in sort(collect(keys(stats)))
        s = stats[g]
        status = s.bad == 0 ? "ok  " : "DIFF"
        @printf("  %s %-40s %6d entries  max|Δ| = %.3e  max rel = %.3e", status, g, s.n, s.abs, s.rel)
        s.bad == 0 || @printf("  (%d over atol, worst %s)", s.bad, s.worst)
        println()
        changed = [t for t in s.types if t[1] != t[2]]
        isempty(changed) || println("       type differs: ", join(("$(t[1]) (v6) vs $(t[2]) (v7)" for t in changed), "; "))
        s.shape || println("       lengths differ")
        ok &= s.bad == 0
        if occursin("free_energy", g)
            names = sort(filter(k -> group(k) == g, collect(keys(v6.values))); by = k -> something(tryparse(Int, strip(replace(k, g => ""), ['[', ']'])), 0))
            fe6 = reduce(vcat, (v6.values[k] for k in names))
            fe7 = reduce(vcat, (v7.values[k] for k in names if haskey(v7.values, k)); init = Float64[])
            @printf("       v6 final = %.12g   v7 final = %.12g\n", last(fe6), isempty(fe7) ? NaN : last(fe7))
        end
    end
    println("  => ", ok ? "PASS" : "FAIL")
    return ok
end

examples = isempty(ARGS) ?
    sort(unique(first(split(f, "_v6.jls")) for f in readdir(RESULTS_DIR) if endswith(f, "_v6.jls") && isfile(joinpath(RESULTS_DIR, replace(f, "_v6.jls" => "_v7.jls"))))) :
    ARGS
results = [compare_example(e) for e in examples]
println()
println(all(results) ? "all examples agree" : "some examples differ")
exit(all(results) ? 0 : 1)
