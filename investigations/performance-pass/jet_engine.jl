# JET optimization analysis of the engine's hot paths: one rule call, a product of messages, the
# variable marginal and a joint marginal. Prints the number of reports per entry and the first few.
#   julia --project=<env> jet_engine.jl <variant>
ENV["BENCH_ONLY"] = "__none__"
const JV = ARGS[1]
empty!(ARGS); push!(ARGS, JV, "jet", "/dev/null")
include(joinpath(@__DIR__, "bench_micro.jl"))
using JET
mods = (ReactiveMP, MessagePassingRulesBase, Rocket)
function show_report(name, r)
    reps = JET.get_reports(r)
    println("== ", name, ": ", length(reps), " report(s)")
    for rep in reps
        s = sprint(show, rep)
        println("   ", first(replace(s, '\n' => ' '), 300))
    end
    return
end
mapping = MessageMapping(NormalMeanVariance, Target{:out}(), Val((:μ, :v)), nothing, DefaultAlgorithm(), nothing, nothing, nothing)
ms = (msg(NormalMeanVariance(1.0, 2.0)), msg(PointMass(0.5)))
show_report("MessageMapping NMV->out (BP)", JET.report_opt(mapping, (typeof(ms), Nothing); target_modules = mods))
v = randomvar(); ctx = MessageProductContext()
msgs = AbstractMessage[msg(NormalMeanVariance(randn(), 1.0)) for _ in 1:3]
show_report("compute_product_of_messages (Vector{AbstractMessage})", JET.report_opt(compute_product_of_messages, (typeof(v), typeof(ctx), typeof(msgs)); target_modules = mods))
l, r = msg(NormalMeanVariance(0.0, 1.0)), msg(NormalMeanVariance(1.0, 1.0))
show_report("compute_product_of_two_messages (concrete)", JET.report_opt(ReactiveMP.compute_product_of_two_messages, (typeof(v), typeof(ctx), typeof(l), typeof(r)); target_modules = mods))
jm = MarginalMapping(NormalMeanVariance, ClusterTarget((:out, :μ)), Val((:out, :μ)), Val((:v,)), DefaultAlgorithm(), nothing)
jms = (msg(NormalMeanVariance(1.0, 2.0)), msg(NormalMeanVariance(0.0, 1.0))); jqs = (mrg(PointMass(0.5)),)
show_report("MarginalMapping NMV (out,μ)", JET.report_opt(ReactiveMP.compute_marginal, (typeof(jm), typeof(jms), typeof(jqs)); target_modules = mods))
