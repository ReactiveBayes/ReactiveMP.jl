const NO_MAIN = true
empty!(ARGS); append!(ARGS, ["X", "ssm1", "p", "/dev/null"])
include("/Users/bvdmitri/Projects/Julia/ReactiveMP.jl/investigations/performance-pass/bench_models.jl")
using BenchmarkTools
import GraphPPL
rng = StableRNG(1); y = randn(rng, 1000)
r = infer(model = ssm1(P = 10.0), data = (y = y,), options = (limit_stack_depth = 500,), free_energy = Float64, session = nothing)
pm = r.model; m = RxInfer.getmodel(pm)
println("nodes: ", length(collect(GraphPPL.factor_nodes(m))), " factors, ", length(collect(GraphPPL.variable_nodes(m))), " variables")
cnt = Ref(0)
b1 = @benchmark GraphPPL.variable_nodes($m) do _, v
    $cnt[] += 1
end
b2 = @benchmark GraphPPL.factor_nodes($m) do _, v
    $cnt[] += 1
end
b3 = @benchmark RxInfer.getrandomvars($pm)
b4 = @benchmark RxInfer.getdatavars($pm)
b5 = @benchmark RxInfer.getfactornodes($pm)
b6 = @benchmark RxInfer.getvardict($m)
b7 = @benchmark RxInfer.create_model(RxInfer.ConditionedModelGenerator($(ssm1(P = 10.0)), (y = $y,)))
for (n, b) in (("variable_nodes pass", b1), ("factor_nodes pass", b2), ("getrandomvars", b3), ("getdatavars", b4), ("getfactornodes", b5), ("getvardict", b6), ("create_model plain GraphPPL (no RxInfer plugins)", b7))
    println(rpad(n, 50), BenchmarkTools.prettytime(minimum(b).time), "  ", minimum(b).allocs, " allocs")
end
