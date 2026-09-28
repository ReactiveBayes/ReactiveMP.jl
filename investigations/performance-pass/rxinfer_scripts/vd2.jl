using RxInfer, BenchmarkTools
import RxInfer: GraphPPL
const NO_MAIN = true
empty!(ARGS); append!(ARGS, ["X", "ssm1", "p", "/dev/null"])
include("/Users/bvdmitri/Projects/Julia/ReactiveMP.jl/investigations/performance-pass/bench_models.jl")
for (label, mdl, dat) in (("tiny", nothing, nothing), ("ssm1 n=1000", ssm1(P = 10.0), (y = randn(1000),)))
    local r
    if mdl === nothing
        @eval @model function m1(y)
            θ ~ Beta(1.0, 1.0)
            y .~ Bernoulli(θ)
        end
        r = infer(model = m1(), data = (y = [1.0, 0.0, 1.0],), session = nothing)
    else
        r = infer(model = mdl, data = dat, options = (limit_stack_depth = 500,), session = nothing)
    end
    m = RxInfer.getmodel(r.model)
    for (n, f) in (("getvardict", RxInfer.getvardict), ("top-level", isdefined(RxInfer, :gettoplevelvardict) ? RxInfer.gettoplevelvardict : (m -> GraphPPL.variables(RxInfer.getvardict(m)))))
        ts = [minimum(@benchmark $f($m)) for _ in 1:3]
        println(rpad("$label $n", 28), minimum(t -> t.time, ts), " ns  ", ts[1].allocs, " allocs")
    end
end
