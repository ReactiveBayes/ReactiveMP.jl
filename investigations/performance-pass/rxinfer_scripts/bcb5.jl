const NO_MAIN = true
empty!(ARGS); append!(ARGS, ["X", "iid", "p", "/dev/null"])
include("/Users/bvdmitri/Projects/Julia/ReactiveMP.jl/investigations/performance-pass/bench_models.jl")
using Profile
Y = 3.0 .+ 0.5 .* randn(StableRNG(42), 100)
INIT = @initialization begin
    q(τ) = GammaShapeRate(1.0, 1.0)
end
g(; it = 1, kw...) = infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = it, free_energy = Float64, session = nothing; kw...)
function allocsites(f)
    f(); f()
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = 1.0 f()
    r = Profile.Allocs.fetch()
    sites = Dict{String, Int}()
    for a in r.allocs
        key = "?"
        for sf in a.stacktrace
            fl = String(sf.file)
            if occursin("RxInfer.jl/src", fl) || occursin("ReactiveMP.jl/src", fl) || occursin("Rocket.jl/src", fl) || occursin("ReactiveMP.jl/lib", fl) || occursin("GraphPPL.jl/src", fl)
                key = string(basename(fl), ":", sf.line, " ", sf.func, " [", (a.type isa DataType ? nameof(a.type) : a.type), "]")
                break
            end
        end
        sites[key] = get(sites, key, 0) + a.size
    end
    return sites
end
println("bytes base ", @allocated(g()), "  bench ", @allocated(g(callbacks = RxInferBenchmarkCallbacks())))
function persite(it, cb)
    return allocsites(() -> cb === nothing ? g(it = it) : g(it = it, callbacks = cb))
end
s1b = persite(1, nothing); s3b = persite(3, nothing)
s1c = persite(1, RxInferBenchmarkCallbacks()); s3c = persite(3, RxInferBenchmarkCallbacks())
println("--- setup extra (it=1, bench - none)")
for (d, k) in sort([(get(s1c, k, 0) - get(s1b, k, 0), k) for k in union(keys(s1b), keys(s1c))]; rev = true)[1:8]
    println(d, "  ", first(k, 160))
end
println("--- per-iteration extra ((it3-it1) bench - none)")
sb = Dict(k => get(s3b, k, 0) - get(s1b, k, 0) for k in union(keys(s3b), keys(s1b)))
sc = Dict(k => get(s3c, k, 0) - get(s1c, k, 0) for k in union(keys(s3c), keys(s1c)))
diffs = sort([(get(sc, k, 0) - get(sb, k, 0), k) for k in union(keys(sb), keys(sc))]; rev = true)
for (d, k) in diffs[1:14]
    println(d, "  ", first(k, 200))
end
flush(stdout)
