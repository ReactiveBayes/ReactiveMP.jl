const NO_MAIN = true
empty!(ARGS); append!(ARGS, ["X", "iid", "p", "/dev/null"])
include("/Users/bvdmitri/Projects/Julia/ReactiveMP.jl/investigations/performance-pass/bench_models.jl")
Y = 3.0 .+ 0.5 .* randn(StableRNG(42), 1000)
INIT = @initialization begin
    q(τ) = GammaShapeRate(1.0, 1.0)
end
g(; kw...) = infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = Float64, session = nothing; kw...)
for (n, kw) in (("none", (;)), ("trace=true", (trace = true,)), ("trace=(:before_iteration,)", (trace = (:before_iteration, :after_iteration),)))
    g(; kw...); g(; kw...)
    r = g(; kw...)
    ts = minimum([(GC.gc(); @elapsed g(; kw...)) for _ in 1:5])
    println(
        rpad(n, 30), @allocated(g(; kw...)) / 1.0e6, " MB  ", round(ts * 1.0e3; digits = 2), " ms  ",
        haskey(r.model.metadata, :trace) ? length(r.model.metadata[:trace].events) : -1, " events"
    )
end
