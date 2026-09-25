# Type analysis of the hot paths, A vs B.
#   julia --project=envA typeinfo.jl <outdir>     (JET is loaded from a stacked env if present)
using ReactiveMP, Rocket, BayesBase, ExponentialFamily, Distributions, StandardMessagePassingRules, LinearAlgebra, InteractiveUtils
using MessagePassingRulesBase
import MessagePassingRulesBase: Target, ClusterTarget, DefaultAlgorithm
import ReactiveMP: MessageMapping, MarginalMapping, rule_arguments, compute_product_of_messages, compute_product_of_two_messages, MessageProductContext
const HASJET = try
    @eval using JET
    true
catch
    false
end

const VARIANT = let lazy = isdefined(ReactiveMP, Symbol("@invoke_callback"))
    ReactiveMP.Message isa UnionAll ? (isdefined(ReactiveMP, :new_message) ? "F" : lazy ? "D" : "A") : isdefined(ReactiveMP, :run_message_rule) ? (lazy ? "E" : "C") : "B"
end
const OUTDIR = ARGS[1]
io = open(joinpath(OUTDIR, "typeinfo_$(VARIANT).txt"), "w")
say(x...) = (println(io, x...); println(x...))

msg(d) = Message(d, false, false)
mrg(d) = Marginal(d, false, false)

cases = [
    ("NMV->out BP", NormalMeanVariance, Target{:out}(), Val((:μ, :v)), nothing, (msg(NormalMeanVariance(1.0, 2.0)), msg(PointMass(0.5))), nothing),
    ("NMP->μ VMP", NormalMeanPrecision, Target{:μ}(), nothing, Val((:out, :τ)), nothing, (mrg(PointMass(0.3)), mrg(GammaShapeRate(2.0, 3.0)))),
]

say("variant ", VARIANT, "; typeof(msg(NMV)) = ", typeof(msg(NormalMeanVariance(1.0, 2.0))))
for (name, fform, target, mn, qn, ms, qs) in cases
    mapping = MessageMapping(fform, target, mn, qn, DefaultAlgorithm(), nothing, nothing, nothing)
    mapping(ms, qs)
    T = (typeof(ms), typeof(qs))
    say("\n== ", name, "   argument types ", T)
    say("return_types(mapping) = ", Base.return_types(mapping, T))
    say("return_types(rule_arguments) = ", Base.return_types(rule_arguments, (typeof(mn), typeof(ms), typeof(qn), typeof(qs))))
    args = rule_arguments(mn, ms, qn, qs)
    say("typeof(args) = ", typeof(args))
    say("return_types(find_message_rule) = ", Base.return_types(MessagePassingRulesBase.find_message_rule, (typeof(fform) <: Type ? Type{fform} : typeof(fform), typeof(target), DefaultAlgorithm, typeof(args))))
    say("-- @code_warntype mapping(ms, qs)")
    b = IOBuffer(); code_warntype(IOContext(b, :color => false), mapping, T); s = String(take!(b))
    println(io, s)
    say("   lines with ::Any = ", count(l -> occursin("::Any", l), split(s, '\n')))
    if HASJET
        r = JET.report_opt(mapping, T)
        reps = JET.get_reports(r)
        say("JET report_opt (runtime dispatch / captured vars): ", length(reps), " reports")
        b = IOBuffer(); show(IOContext(b, :color => false), r); println(io, String(take!(b)))
    end
end

# The product at a variable: messages arrive as Vector{AbstractMessage}
v = randomvar(); ctx = MessageProductContext()
msgs = AbstractMessage[msg(NormalMeanVariance(0.0, 1.0)) for _ in 1:3]
say("\n== compute_product_of_messages(::RandomVariable, ctx, ::Vector{AbstractMessage})")
say("return_types = ", Base.return_types(compute_product_of_messages, (typeof(v), typeof(ctx), typeof(msgs))))
m1 = msgs[1]
say("return_types two(Message, Message) = ", Base.return_types(compute_product_of_two_messages, (typeof(v), typeof(ctx), typeof(m1), typeof(m1))))
if HASJET
    r = JET.report_opt(compute_product_of_messages, (typeof(v), typeof(ctx), typeof(msgs)))
    say("JET report_opt product: ", length(JET.get_reports(r)), " reports")
    b = IOBuffer(); show(IOContext(b, :color => false), r); println(io, String(take!(b)))
    r = JET.report_opt(compute_product_of_two_messages, (typeof(v), typeof(ctx), typeof(m1), typeof(m1)))
    say("JET report_opt product of two (concrete Message types): ", length(JET.get_reports(r)), " reports")
    b = IOBuffer(); show(IOContext(b, :color => false), r); println(io, String(take!(b)))
end
if VARIANT == "C"
    say("\n== inside the kernels, at the concrete data types")
    for (name, fform, target, mn, qn, ms, qs) in cases
        mapping = MessageMapping(fform, target, mn, qn, DefaultAlgorithm(), nothing, nothing, nothing)
        md, qd = ReactiveMP.data_of(ms), ReactiveMP.data_of(qs)
        T = (typeof(mapping), typeof(md), typeof(qd), typeof(ReactiveMP.annotations_of(ms)), typeof(ReactiveMP.annotations_of(qs)), ReactiveMP.AnnotationDict)
        say(name, ": return_types(run_message_rule) = ", Base.return_types(ReactiveMP.run_message_rule, T))
        say(name, ": return_types(kernel_arguments) = ", Base.return_types(ReactiveMP.kernel_arguments, (typeof(mn), typeof(md), typeof(qn), typeof(qd))))
        if HASJET
            r = JET.report_opt(ReactiveMP.run_message_rule, T)
            say(name, ": JET report_opt run_message_rule: ", length(JET.get_reports(r)), " reports")
            b = IOBuffer(); show(IOContext(b, :color => false), r); println(io, String(take!(b)))
        end
    end
    a = NormalMeanVariance(0.0, 1.0); ann = ReactiveMP.AnnotationDict()
    if HASJET
        r = JET.report_opt(ReactiveMP.product_kernel, (typeof(v), typeof(ctx), typeof(a), typeof(a), typeof(ann), typeof(ann)))
        say("JET report_opt product_kernel(NMV, NMV): ", length(JET.get_reports(r)), " reports")
    end
end
close(io)
