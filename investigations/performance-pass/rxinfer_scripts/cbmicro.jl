using RxInfer, ReactiveMP, BenchmarkTools, MessagePassingRulesBase
import MessagePassingRulesBase: Target, DefaultAlgorithm
import ReactiveMP: MessageMapping, MessageProductContext, compute_product_of_messages
msg(d) = Message(d, false, false)
v = randomvar()
msgs = AbstractMessage[msg(NormalMeanVariance(randn(), 1.0 + rand())) for _ in 1:10]
ms = (msg(NormalMeanVariance(1.0, 2.0)), msg(PointMass(0.5)))
for (n, cb) in (("nothing", nothing), ("(;)", (;)), ("RxInferBenchmarkCallbacks()", RxInferBenchmarkCallbacks()), ("merged(nothing, bench)", ReactiveMP.merge_callbacks(nothing, RxInferBenchmarkCallbacks())))
    ctx = MessageProductContext(callbacks = cb)
    b1 = @benchmark compute_product_of_messages($v, $ctx, $msgs)
    mapping = MessageMapping(NormalMeanVariance, Target{:out}(), Val((:μ, :v)), nothing, DefaultAlgorithm(), nothing, nothing, cb)
    mapping(ms, nothing)
    b2 = @benchmark $mapping($ms, nothing)
    println(rpad(n, 30), "product x10: ", round(minimum(b1).time; digits = 1), " ns ", minimum(b1).allocs, " allocs | rule call: ", round(minimum(b2).time; digits = 1), " ns ", minimum(b2).allocs, " allocs")
end
