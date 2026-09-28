const NO_MAIN = true
empty!(ARGS); append!(ARGS, ["X", "iid", "p", "/dev/null"])
include("/Users/bvdmitri/Projects/Julia/ReactiveMP.jl/investigations/performance-pass/bench_models.jl")
Y = 3.0 .+ 0.5 .* randn(StableRNG(42), 1000)
INIT = @initialization begin
    q(τ) = GammaShapeRate(1.0, 1.0)
end
g(it; kw...) = infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = it, free_energy = Float64, session = nothing; kw...)
for it in (1, 10, 20), (n, kw) in (("none", (;)), ("bench", (callbacks = RxInferBenchmarkCallbacks(),)), ("namedtuple", (callbacks = (before_iteration = e -> nothing,),)))
    g(it; kw...); g(it; kw...)
    println("it=$it ", rpad(n, 12), @allocated(g(it; kw...)) / 1.0e6, " MB")
end
y2 = cumsum(randn(StableRNG(1), 1000))
for (n, kw) in (("none", (;)), ("bench", (callbacks = RxInferBenchmarkCallbacks(),)))
    h() = infer(model = ssm1(P = 10.0), data = (y = y2,), options = (limit_stack_depth = 500,), session = nothing; kw...)
    h(); h()
    println("ssm1 ", rpad(n, 12), @allocated(h()) / 1.0e6, " MB")
end
