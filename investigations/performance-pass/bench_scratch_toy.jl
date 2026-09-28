# The engine's cost of an untyped scratch, isolated: two identical cheap rules with a scratch, one
# declaring `scratch_type`. Run in an environment whose base package has `scratch_type` (TS).
using ReactiveMP, BayesBase, ExponentialFamily, BenchmarkTools, MessagePassingRulesBase
import MessagePassingRulesBase: Target, DefaultAlgorithm
struct ToyNode end
@define_factor_node(node = ToyNode, type = Stochastic, interfaces = [:out, :a, :b])
@define_message_update_rule(
    node = ToyNode, target = :out,
    args = (m[:a]::NormalMeanVariance, m[:b]::PointMass),
    scratch = (args) -> (acc = zeros(2),),
    body = (scratch, args) -> begin
        scratch.acc[1] = mean(args.m[:a]) + mean(args.m[:b]); scratch.acc[2] = var(args.m[:a])
        NormalMeanVariance(scratch.acc[1], scratch.acc[2])
    end,
)
@define_message_update_rule(
    node = ToyNode, target = :a,
    args = (m[:out]::NormalMeanVariance, m[:b]::PointMass),
    scratch = (args) -> (acc = zeros(2),),
    scratch_type = (args) -> @NamedTuple{acc::Vector{Float64}},
    body = (scratch, args) -> begin
        scratch.acc[1] = mean(args.m[:out]) - mean(args.m[:b]); scratch.acc[2] = var(args.m[:out])
        NormalMeanVariance(scratch.acc[1], scratch.acc[2])
    end,
)
msg(d) = Message(d, false, false)
ms = (msg(NormalMeanVariance(1.0, 2.0)), msg(PointMass(0.5)))
untyped = ReactiveMP.MessageMapping(ToyNode, Target{:out}(), Val((:a, :b)), nothing, DefaultAlgorithm(), nothing, nothing, nothing)
typed = ReactiveMP.MessageMapping(ToyNode, Target{:a}(), Val((:out, :b)), nothing, DefaultAlgorithm(), nothing, nothing, nothing)
untyped(ms, nothing); typed(ms, nothing)
for (name, f) in (("scratch, no scratch_type", untyped), ("scratch with scratch_type", typed))
    b = @benchmark $f($ms, nothing)
    println(rpad(name, 28), round(minimum(b).time; digits = 1), " ns  ", minimum(b).allocs, " allocs  ", minimum(b).memory, " B")
end
