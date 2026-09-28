# RxInfer-level option costs on one model: session, free_energy type, callbacks (benchmark/trace),
# returnvars keep-each vs keep-last, and infer's return-type inference.
#   julia --project=<env> bench_rxinfer_opts.jl <variant> <tag> <outfile>
const NO_MAIN = true
const VAR_, TAG_, OUT_ = ARGS[1], ARGS[2], ARGS[3]
empty!(ARGS); append!(ARGS, [VAR_, "iid", TAG_, "/dev/null"])
include(joinpath(@__DIR__, "bench_models.jl"))
using Statistics

rng = StableRNG(42)
const Y = 3.0 .+ 0.5 .* randn(rng, 1000)
const Yssm = cumsum(randn(rng, 1000)) .+ sqrt(10.0) .* randn(rng, 1000)
const INIT = @initialization begin
    q(τ) = GammaShapeRate(1.0, 1.0)
end

cases = [
    ("iid base (session=nothing, fe=Float64)", () -> infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = Float64, session = nothing)),
    ("iid session=default", () -> infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = Float64)),
    ("iid free_energy=true", () -> infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = true, session = nothing)),
    ("iid free_energy=false", () -> infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = false, session = nothing)),
    ("iid benchmark=true", () -> infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = Float64, session = nothing, benchmark = true)),
    ("iid trace=true", () -> infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = Float64, session = nothing, trace = true)),
    ("iid returnvars=KeepEach", () -> infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = Float64, session = nothing, returnvars = KeepEach())),
    ("iid returnvars=(μ=KeepLast,)", () -> infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = Float64, session = nothing, returnvars = (μ = KeepLast(),))),
    ("iid all defaults (session, fe=true)", () -> infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = true)),
    ("ssm1 base", () -> infer(model = ssm1(P = 10.0), data = (y = Yssm,), options = (limit_stack_depth = 500,), free_energy = Float64, session = nothing)),
    ("ssm1 no FE", () -> infer(model = ssm1(P = 10.0), data = (y = Yssm,), options = (limit_stack_depth = 500,), session = nothing)),
    ("ssm1 all defaults (session, fe=true)", () -> infer(model = ssm1(P = 10.0), data = (y = Yssm,), options = (limit_stack_depth = 500,), free_energy = true)),
    ("ssm1 session=default fe=Float64", () -> infer(model = ssm1(P = 10.0), data = (y = Yssm,), options = (limit_stack_depth = 500,), free_energy = Float64)),
]

for (name, f) in cases
    f(); f()
end
rounds = parse(Int, get(ENV, "BENCH_ROUNDS", "3"))
res = Dict(name => Float64[] for (name, _) in cases)
bytes = Dict(name => Int[] for (name, _) in cases)
for r in 1:rounds, (name, f) in (isodd(r) ? cases : reverse(cases))
    s = [(GC.gc(); @timed f()) for _ in 1:3]
    push!(res[name], minimum(x -> x.time, s)); push!(bytes[name], minimum(x -> x.bytes, s))
end
open(OUT_, "a") do io
    for (name, _) in cases
        l = join((VAR_, TAG_, name, median(res[name]), minimum(res[name]), minimum(bytes[name])), '\t')
        println(l); println(io, l)
    end
end
# return type inference of infer
rt = Base.return_types(RxInfer.infer, Tuple{})
println("infer return types (kwcall, iid): ", Base.infer_return_type(() -> infer(model = iid(), data = (y = Y,), constraints = MeanField(), initialization = INIT, iterations = 10, free_energy = Float64, session = nothing)))
