# JET optimisation analysis of RxInfer's own code on ssm1, iid, and the streaming filter.
const NO_MAIN = true
const VAR_ = ARGS[1]
empty!(ARGS); append!(ARGS, [VAR_, "iid", "jet", "/dev/null"])
include(joinpath(@__DIR__, "bench_models.jl"))
using JET
rng = StableRNG(42)
Y = 3.0 .+ 0.5 .* randn(rng, 1000)
Yssm = cumsum(randn(rng, 200)) .+ sqrt(10.0) .* randn(rng, 200)
INIT = @initialization begin
    q(τ) = GammaShapeRate(1.0, 1.0)
end
f_iid() = infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = Float64, session = nothing)
f_ssm() = infer(model = ssm1(P = 10.0), data = (y = Yssm,), options = (limit_stack_depth = 500,), free_energy = Float64, session = nothing)
au = @autoupdates begin
    x_prev_mean, x_prev_var = mean_var(q(x))
end
INITF = @initialization begin
    q(x) = NormalMeanVariance(0.0, 1.0e3)
end
f_filter() = infer(model = kfilter(), datastream = from(Yssm) |> map(NamedTuple{(:y,), Tuple{Float64}}, d -> (y = d,)), autoupdates = au, initialization = INITF, keephistory = 200, historyvars = (x = KeepLast(),), autostart = true, free_energy = Float64, session = nothing)
b_iid() = RxInfer.batch_inference(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = Float64)
b_ssm() = RxInfer.batch_inference(model = ssm1(P = 10.0), data = (y = Yssm,), options = (limit_stack_depth = 500,), free_energy = Float64)
s_filter() = RxInfer.streaming_inference(model = kfilter(), datastream = from(Yssm) |> map(NamedTuple{(:y,), Tuple{Float64}}, d -> (y = d,)), autoupdates = au, initialization = INITF, keephistory = 200, historyvars = (x = KeepLast(),), autostart = true, free_energy = Float64)
f_iid(); f_ssm(); f_filter(); b_iid(); b_ssm(); s_filter()
for (name, f) in (("iid", f_iid), ("ssm1", f_ssm), ("filter", f_filter), ("batch_iid", b_iid), ("batch_ssm1", b_ssm), ("streaming_filter", s_filter))
    r = JET.report_opt(f, (); target_modules = (RxInfer,))
    reps = JET.get_reports(r)
    println("\n==== $name: $(length(reps)) reports (target RxInfer)")
    open(joinpath(@__DIR__, "results", "rxinfer_jet_$(VAR_)_$(name).txt"), "w") do io
        show(IOContext(io, :limit => false), r)
    end
    # compact: one line per report, innermost RxInfer frame + kind
    for rep in reps
        vst = rep.vst
        fr = last(vst)
        println(nameof(typeof(rep)), "  @ ", fr.file, ":", fr.line, "  ", fr.linfo.def isa Method ? fr.linfo.def.name : fr.linfo.def)
    end
end
