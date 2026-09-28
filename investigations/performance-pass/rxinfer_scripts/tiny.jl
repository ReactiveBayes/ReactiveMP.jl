using RxInfer, BenchmarkTools
@model function m1(y)
    θ ~ Beta(1.0, 1.0)
    y .~ Bernoulli(θ)
end
d = (y = [1.0, 0.0, 1.0],)
f0() = infer(model = m1(), data = d, session = nothing)
f1() = infer(model = m1(), data = d)
f2() = infer(model = m1(), data = d, session = nothing, free_energy = true)
f3() = infer(model = m1(), data = d, session = nothing, free_energy = Float64)
f0(); f1(); f2(); f3()
for (n, f) in (("tiny session=nothing", f0), ("tiny session=default", f1), ("tiny fe=true", f2), ("tiny fe=Float64", f3))
    b = @benchmark $f() samples = 2000 seconds = 8
    println(rpad(n, 26), BenchmarkTools.prettytime(minimum(b).time), "  median ", BenchmarkTools.prettytime(median(b).time), "  ", minimum(b).allocs, " allocs ", minimum(b).memory, " B")
end
