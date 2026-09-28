# End-to-end inference through RxInfer, one model per fresh process, on v6 (RxInfer 5.5.2 over
# ReactiveMP 6.5.0) or v7 (RxInfer's refactor/reactivemp-v7 over this repository).
#   julia --project=<env> bench_models.jl <variant> <model> <tag> <outfile> [<posteriors-dir>]
# Records: package load; time to first inference (compile included); model creation alone
# (GraphPPL + plugins + ReactiveMP nodes and activation, as `batch_inference` does it); steady
# `infer` min/median time, bytes and GC share; for iterative models also at 2x iterations, so the
# per-iteration cost is (T(2I) - T(I)) / I. Always `session = nothing`; `free_energy = Float64`
# unless the model name ends in `+defaults`, which uses `free_energy = true` and the session.

const T0 = time()
using RxInfer, StableRNGs, LinearAlgebra, Statistics, Serialization
import ReactiveMP
const V7 = pkgversion(ReactiveMP).major >= 7
@static if V7
    using DeltaMessagePassingRules, DiscreteTransitionMessagePassingRules
end
const TLOAD = time() - T0

const VARIANT = ARGS[1]
const MODELARG = ARGS[2]
const TAG = ARGS[3]
const OUT = ARGS[4]
const PDIR = get(ARGS, 5, "")
const DEFAULTS = endswith(MODELARG, "+defaults")
const MODEL = replace(MODELARG, "+defaults" => "")

# (ssm1) univariate Kalman smoother, BP
@model function ssm1(y, P)
    x_prior ~ Normal(mean = 0.0, variance = 10000.0)
    x_prev = x_prior
    for i in eachindex(y)
        x[i] ~ Normal(mean = x_prev, variance = 1.0)
        y[i] ~ Normal(mean = x[i], variance = P)
        x_prev = x[i]
    end
end

# (ssm2) 2-d rotating Kalman smoother, BP with matrix products
@model function ssm2(y, A, Q, R)
    x_prior ~ MvNormal(mean = zeros(2), covariance = 100.0 * diageye(2))
    x_prev = x_prior
    for i in eachindex(y)
        x[i] ~ MvNormal(mean = A * x_prev, covariance = Q)
        y[i] ~ MvNormal(mean = x[i], covariance = R)
        x_prev = x[i]
    end
end

# (iid) Normal with unknown mean and precision, mean-field VMP
@model function iid(y)
    μ ~ Normal(mean = 0.0, variance = 100.0)
    τ ~ Gamma(shape = 1.0, rate = 1.0)
    for i in eachindex(y)
        y[i] ~ Normal(mean = μ, precision = τ)
    end
end

# (betabern) Beta-Bernoulli, exact
@model function betabern(y)
    θ ~ Beta(1.0, 1.0)
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

# (nl) nonlinear state space model, Delta node with Linearization
f_nl(x) = x + 0.1 * sin(x)
@static if V7
    @model function nl(y)
        x_prior ~ Normal(mean = 0.0, variance = 10.0)
        x_prev = x_prior
        for i in eachindex(y)
            x[i] ~ Normal(mean = x_prev, variance = 0.1)
            z[i] := f_nl(x[i]) where {algorithm = DeltaApproximation(method = Linearization())}
            y[i] ~ Normal(mean = z[i], variance = 0.5)
            x_prev = x[i]
        end
    end
else
    @model function nl(y)
        x_prior ~ Normal(mean = 0.0, variance = 10.0)
        x_prev = x_prior
        for i in eachindex(y)
            x[i] ~ Normal(mean = x_prev, variance = 0.1)
            z[i] := f_nl(x[i]) where {meta = DeltaMeta(method = Linearization())}
            y[i] ~ Normal(mean = z[i], variance = 0.5)
            x_prev = x[i]
        end
    end
end

# (hmm) hidden Markov model, DiscreteTransition, structured VMP
@model function hmm(x)
    A ~ DirichletCollection(ones(3, 3))
    B ~ DirichletCollection([10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0])
    s_0 ~ Categorical(fill(1.0 / 3.0, 3))
    s_prev = s_0
    for t in eachindex(x)
        s[t] ~ DiscreteTransition(s_prev, A)
        x[t] ~ DiscreteTransition(s[t], B)
        s_prev = s[t]
    end
end

# (gmm) univariate two-component Gaussian mixture, mean-field
@model function gmm(y)
    s ~ Beta(1.0, 1.0)
    m1 ~ Normal(mean = -2.0, variance = 1.0e3)
    w1 ~ Gamma(shape = 0.01, rate = 0.01)
    m2 ~ Normal(mean = 2.0, variance = 1.0e3)
    w2 ~ Gamma(shape = 0.01, rate = 0.01)
    for i in eachindex(y)
        z[i] ~ Bernoulli(s)
        y[i] ~ NormalMixture(switch = z[i], m = (m1, m2), p = (w1, w2))
    end
end

# (linreg) Bayesian linear regression, BP through `*` and `+`
@model function linreg(x, y)
    a ~ Normal(mean = 0.0, variance = 1.0)
    b ~ Normal(mean = 0.0, variance = 100.0)
    for i in eachindex(y)
        y[i] ~ Normal(mean = x[i] * b + a, variance = 1.0)
    end
end

# (filter) streaming Kalman filter with @autoupdates
@model function kfilter(y, x_prev_mean, x_prev_var)
    x_prev ~ Normal(mean = x_prev_mean, variance = x_prev_var)
    x ~ Normal(mean = x_prev, variance = 1.0)
    y ~ Normal(mean = x, variance = 10.0)
end

function onehot(rng, p)
    s = zeros(length(p)); s[rand(rng, Categorical(p))] = 1.0
    return s
end

sizeof_model(default) = (m = match(r"@(\d+)$", MODEL); m === nothing ? default : parse(Int, m[1]))
basename_model() = replace(MODEL, r"@\d+$" => "")

fe_opt() = DEFAULTS ? true : Float64
session_opt() = DEFAULTS ? RxInfer.default_session() : nothing

# returns (run(iterations), iterations, creation thunk or nothing)
function setup()
    rng = StableRNG(42)
    name = basename_model()
    if name == "ssm1"
        n = sizeof_model(1000)
        y = cumsum(randn(rng, n)) .+ sqrt(10.0) .* randn(rng, n)
        return (iters -> infer(model = ssm1(P = 10.0), data = (y = y,), options = (limit_stack_depth = 500,), free_energy = fe_opt(), session = session_opt())), 0
    elseif name == "ssm2"
        n = sizeof_model(1000); θ = π / 20
        A = [cos(θ) -sin(θ); sin(θ) cos(θ)]; Q = diageye(2); R = 5.0 * diageye(2)
        xs = Vector{Vector{Float64}}(undef, n); ys = similar(xs); xp = [10.0, -10.0]
        for i in 1:n
            xp = A * xp .+ randn(rng, 2); xs[i] = xp; ys[i] = xp .+ sqrt(5.0) .* randn(rng, 2)
        end
        return (iters -> infer(model = ssm2(A = A, Q = Q, R = R), data = (y = ys,), options = (limit_stack_depth = 500,), free_energy = fe_opt(), session = session_opt())), 0
    elseif name == "iid"
        n = sizeof_model(1000)
        y = 3.0 .+ 0.5 .* randn(rng, n)
        init = @initialization begin
            q(τ) = GammaShapeRate(1.0, 1.0)
        end
        return (iters -> infer(model = iid(), data = (y = y,), constraints = MeanField(), initialization = init, iterations = iters, free_energy = fe_opt(), session = session_opt())), 10
    elseif name == "betabern"
        n = sizeof_model(5000)
        y = float.(rand(rng, n) .< 0.3)
        return (iters -> infer(model = betabern(), data = (y = y,), free_energy = fe_opt(), session = session_opt())), 0
    elseif name == "nl"
        n = sizeof_model(300)
        xs = zeros(n); xp = 0.0
        for i in 1:n
            xp = f_nl(xp) + sqrt(0.1) * randn(rng); xs[i] = xp
        end
        y = f_nl.(xs) .+ sqrt(0.5) .* randn(rng, n)
        init = @initialization begin
            μ(x) = NormalMeanVariance(0.0, 10.0)
        end
        return (iters -> infer(model = nl(), data = (y = y,), initialization = init, iterations = iters, options = (limit_stack_depth = 500,), free_energy = fe_opt(), session = session_opt())), 5
    elseif name == "hmm"
        n = sizeof_model(300)
        At = [0.9 0.0 0.1; 0.1 0.9 0.0; 0.0 0.1 0.9]; Bt = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9]
        s = [1.0, 0.0, 0.0]; x = Vector{Vector{Float64}}(undef, n)
        for t in 1:n
            a = At * s; s = onehot(rng, a ./ sum(a)); b = Bt * s; x[t] = onehot(rng, b ./ sum(b))
        end
        cons = @constraints begin
            q(s, s_0, A, B) = q(s, s_0)q(A)q(B)
        end
        init = @initialization begin
            q(A) = vague(DirichletCollection, (3, 3))
            q(B) = vague(DirichletCollection, (3, 3))
            q(s) = vague(Categorical, 3)
        end
        return (
                iters -> infer(
                    model = hmm(), data = (x = x,), constraints = cons, initialization = init, iterations = iters,
                    options = (limit_stack_depth = 500,), free_energy = fe_opt(), session = session_opt()
                )
            ), 10
    elseif name == "gmm"
        n = sizeof_model(500)
        z = rand(rng, n) .< 0.4
        y = [zi ? -2.0 + 0.5 * randn(rng) : 3.0 + 0.8 * randn(rng) for zi in z]
        init = @initialization begin
            q(s) = vague(Beta)
            q(m1) = NormalMeanVariance(-2.0, 1.0e3)
            q(m2) = NormalMeanVariance(2.0, 1.0e3)
            q(w1) = vague(GammaShapeRate)
            q(w2) = vague(GammaShapeRate)
        end
        return (iters -> infer(model = gmm(), data = (y = y,), constraints = MeanField(), initialization = init, iterations = iters, free_energy = fe_opt(), session = session_opt())), 10
    elseif name == "linreg"
        n = sizeof_model(1000)
        x = randn(rng, n); y = 1.5 .+ 0.7 .* x .+ randn(rng, n)
        init = @initialization begin
            μ(b) = NormalMeanVariance(0.0, 100.0)
        end
        return (iters -> infer(model = linreg(), data = (x = x, y = y), initialization = init, iterations = iters, free_energy = fe_opt(), session = session_opt())), 10
    elseif name == "filter"
        n = sizeof_model(1000)
        y = cumsum(randn(rng, n)) .+ sqrt(10.0) .* randn(rng, n)
        au = @autoupdates begin
            x_prev_mean, x_prev_var = mean_var(q(x))
        end
        init = @initialization begin
            q(x) = NormalMeanVariance(0.0, 1.0e3)
        end
        return (
                iters -> infer(
                    model = kfilter(), datastream = from(y) |> map(NamedTuple{(:y,), Tuple{Float64}}, d -> (y = d,)), autoupdates = au,
                    initialization = init, keephistory = n, historyvars = (x = KeepLast(),), autostart = true, free_energy = fe_opt(), session = session_opt()
                )
            ), 0
    end
    error("unknown model $MODEL")
end

summarise(d) = d isa AbstractVector ? map(summarise, d) : summarise1(d)
function summarise1(d)
    m = try
        mean(d)
    catch
        nothing
    end
    c = try
        (d isa Union{Categorical, DirichletCollection} ? nothing : cov(d))
    catch
        try
            var(d)
        catch
            nothing
        end
    end
    return (nameof(typeof(d)), m, c)
end

posteriors_of(result) = hasproperty(result, :posteriors) ? result.posteriors : result.history
free_energy_of(result) = hasproperty(result, :free_energy) ? result.free_energy : (hasproperty(result, :free_energy_history) ? result.free_energy_history : nothing)

function line(stage, tmin, tmed, bytes, gc = 0.0)
    return join((VARIANT, TAG, MODELARG, stage, tmin, tmed, bytes, gc), '\t')
end

function main()
    run, iters = setup()
    it1 = max(iters, 1)
    ttfx = @elapsed result = run(it1)
    if !isempty(PDIR)
        post = Dict(k => summarise(v) for (k, v) in pairs(posteriors_of(result)))
        serialize(joinpath(PDIR, "$(VARIANT)_$(MODELARG)_$(TAG).jls"), (post, free_energy_of(result)))
    end
    GC.gc()
    nsamples = parse(Int, get(ENV, "BENCH_SAMPLES", "7"))
    samples = [(GC.gc(); @timed run(it1)) for _ in 1:nsamples]
    ts = map(s -> s.time, samples); bs = map(s -> s.bytes, samples); gcs = map(s -> s.gctime / s.time, samples)
    lines = String[]
    push!(lines, line("load", TLOAD, TLOAD, 0))
    push!(lines, line("ttfx", ttfx, ttfx, 0))
    push!(lines, line("infer(I=$it1)", minimum(ts), median(ts), minimum(bs), median(gcs)))
    if iters > 0
        s2 = [(GC.gc(); @timed run(2iters)) for _ in 1:nsamples]
        t2 = map(s -> s.time, s2); b2 = map(s -> s.bytes, s2)
        push!(lines, line("infer(I=$(2iters))", minimum(t2), median(t2), minimum(b2), median(map(s -> s.gctime / s.time, s2))))
        push!(lines, line("per-iteration", (minimum(t2) - minimum(ts)) / iters, (median(t2) - median(ts)) / iters, (minimum(b2) - minimum(bs)) / iters))
        push!(lines, line("setup (T(I) - I*per-iteration)", minimum(ts) - iters * (minimum(t2) - minimum(ts)) / iters, NaN, minimum(bs) - (minimum(b2) - minimum(bs))))
    end
    foreach(println, lines)
    return open(io -> foreach(l -> println(io, l), lines), OUT, "a")
end
isdefined(Main, :NO_MAIN) || main()
