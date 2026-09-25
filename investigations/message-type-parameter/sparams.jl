using ReactiveMP, BayesBase, ExponentialFamily, MessagePassingRulesBase, StandardMessagePassingRules, InteractiveUtils
import MessagePassingRulesBase: Target, DefaultAlgorithm
mapping = ReactiveMP.MessageMapping(NormalMeanVariance, Target{:out}(), Val((:μ, :v)), nothing, DefaultAlgorithm(), nothing, nothing, nothing)
ms = (Message(NormalMeanVariance(1.0, 2.0), false, false), Message(PointMass(0.5), false, false))
function show_sparams(f, T, label)
    ci = only(code_typed(f, T; optimize = true)).first
    lines = filter(l -> occursin("_compute_sparams", l) || occursin("body", l), split(sprint(show, ci), '\n'))
    println(label, ": ", count(l -> occursin("_compute_sparams", l), lines), " _compute_sparams sites")
    return foreach(l -> println("   ", strip(l)[1:min(end, 220)]), filter(l -> occursin("_compute_sparams", l), lines))
end
show_sparams(mapping, (typeof(ms), Nothing), "mapping")
v = randomvar(); ctx = ReactiveMP.MessageProductContext()
show_sparams(ReactiveMP.compute_product_of_messages, (typeof(v), typeof(ctx), Vector{AbstractMessage}), "product of messages")
qm = ReactiveMP.MarginalMapping(NormalMeanVariance, MessagePassingRulesBase.ClusterTarget((:out, :μ)), Val((:out, :μ)), Val((:v,)), DefaultAlgorithm(), nothing)
if isdefined(ReactiveMP, :run_message_rule)
    md = ReactiveMP.data_of(ms); ma = ReactiveMP.annotations_of(ms)
    show_sparams(ReactiveMP.run_message_rule, (typeof(mapping), typeof(md), Nothing, typeof(ma), Nothing, ReactiveMP.AnnotationDict), "run_message_rule")
end
qms = (Message(NormalMeanVariance(1.0, 2.0), false, false), Message(NormalMeanVariance(0.0, 1.0), false, false)); qqs = (Marginal(PointMass(0.5), false, false),)
show_sparams((m, a, b) -> m((a, b)), (typeof(qm), typeof(qms), typeof(qqs)), "MarginalMapping call (via compute)")
opts = ReactiveMP.RandomVariableActivationOptions()
show_sparams(ReactiveMP._compute_marginal_from_messages, (typeof(v), typeof(opts), Vector{AbstractMessage}), "marginal at variable")
show_sparams(ReactiveMP.as_marginal, (Message,), "as_marginal(::Message)")
