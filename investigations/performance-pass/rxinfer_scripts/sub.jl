using RxInfer, BenchmarkTools
@model function step(x_prev, x, y)
    z ~ Normal(mean = x_prev, variance = 1.0)
    x ~ Normal(mean = z, variance = 0.1)
    y ~ Normal(mean = x, variance = 1.0)
end
@model function subchain(y)
    x0 ~ Normal(mean = 0.0, variance = 100.0)
    xp = x0
    for i in eachindex(y)
        x[i] ~ step(x_prev = xp, y = y[i])
        xp = x[i]
    end
end
y = randn(500)
f() = infer(model = subchain(), data = (y = y,), options = (limit_stack_depth = 500,), session = nothing)
f()
m = RxInfer.getmodel(f().model)
println("getvardict:        ", minimum(@benchmark RxInfer.getvardict($m)).time / 1.0e3, " µs")
isdefined(RxInfer, :gettoplevelvardict) && println("gettoplevelvardict: ", minimum(@benchmark RxInfer.gettoplevelvardict($m)).time / 1.0e3, " µs")
println("infer:             ", minimum(@benchmark f() samples = 20 seconds = 20).time / 1.0e6, " ms")
