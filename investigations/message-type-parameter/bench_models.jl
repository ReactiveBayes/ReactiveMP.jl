# End-to-end inference through RxInfer, one model per fresh process.
#   julia --project=envA bench_models.jl <model> <tag> <outfile> <posteriors-dir>
# Records: time to first inference (compile included), then steady-state min/median time and
# min bytes over repeated `infer` calls; for iterative models also at 2x iterations, so the
# per-iteration cost is (T(2I) - T(I)) / I. Dumps posteriors for an exact A/B comparison.

const T0 = time()
using RxInfer, StableRNGs, LinearAlgebra, Statistics, Serialization
using DeltaMessagePassingRules, DiscreteTransitionMessagePassingRules
const TLOAD = time() - T0

const MODEL = ARGS[1]
const TAG = ARGS[2]
const OUT = ARGS[3]
const PDIR = ARGS[4]
const VARIANT = let lazy = isdefined(ReactiveMP, Symbol("@invoke_callback"))
    ReactiveMP.Message isa UnionAll ? (isdefined(ReactiveMP, :new_message) ? "F" : lazy ? "D" : "A") : isdefined(ReactiveMP, :run_message_rule) ? (lazy ? "E" : "C") : "B"
end

# (a1) univariate Kalman smoother, BP
@model function ssm1(y, P)
    x_prior ~ Normal(mean = 0.0, variance = 10000.0)
    x_prev = x_prior
    for i in eachindex(y)
        x[i] ~ Normal(mean = x_prev, variance = 1.0)
        y[i] ~ Normal(mean = x[i], variance = P)
        x_prev = x[i]
    end
end

# (a2) 2-d rotating Kalman smoother, BP with matrix products
@model function ssm2(y, A, Q, R)
    x_prior ~ MvNormal(mean = zeros(2), covariance = 100.0 * diageye(2))
    x_prev = x_prior
    for i in eachindex(y)
        x[i] ~ MvNormal(mean = A * x_prev, covariance = Q)
        y[i] ~ MvNormal(mean = x[i], covariance = R)
        x_prev = x[i]
    end
end

# (b) iid Normal with unknown mean and precision, mean-field VMP
@model function iid(y)
    μ ~ Normal(mean = 0.0, variance = 100.0)
    τ ~ Gamma(shape = 1.0, rate = 1.0)
    for i in eachindex(y)
        y[i] ~ Normal(mean = μ, precision = τ)
    end
end

# (c) nonlinear state space model, Delta node with Linearization
f_nl(x) = x + 0.1 * sin(x)
@model function nl(y)
    x_prior ~ Normal(mean = 0.0, variance = 10.0)
    x_prev = x_prior
    for i in eachindex(y)
        x[i] ~ Normal(mean = x_prev, variance = 0.1)
        z[i] := f_nl(x[i]) where {algorithm = Linearization()}
        y[i] ~ Normal(mean = z[i], variance = 0.5)
        x_prev = x[i]
    end
end

# (d) hidden Markov model, DiscreteTransition, structured VMP
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

function onehot(rng, p)
    s = zeros(length(p)); s[rand(rng, Categorical(p))] = 1.0
    return s
end

function setup(model)
    rng = StableRNG(42)
    if model == "ssm1"
        n = 1000
        y = cumsum(randn(rng, n)) .+ sqrt(10.0) .* randn(rng, n)
        return (iters -> infer(model = ssm1(P = 10.0), data = (y = y,), free_energy = true)), 0
    elseif model == "ssm2"
        n = 1000; θ = π / 20
        A = [cos(θ) -sin(θ); sin(θ) cos(θ)]; Q = diageye(2); R = 5.0 * diageye(2)
        xs = Vector{Vector{Float64}}(undef, n); ys = similar(xs); xp = [10.0, -10.0]
        for i in 1:n
            xp = A * xp .+ randn(rng, 2); xs[i] = xp; ys[i] = xp .+ sqrt(5.0) .* randn(rng, 2)
        end
        return (iters -> infer(model = ssm2(A = A, Q = Q, R = R), data = (y = ys,), options = (limit_stack_depth = 500,), free_energy = true)), 0
    elseif model == "iid"
        y = 3.0 .+ 0.5 .* randn(rng, 1000)
        init = @initialization begin
            q(τ) = GammaShapeRate(1.0, 1.0)
        end
        return (iters -> infer(model = iid(), data = (y = y,), constraints = MeanField(), initialization = init, iterations = iters, free_energy = true)), 10
    elseif model == "nl"
        n = 300
        xs = zeros(n); xp = 0.0
        for i in 1:n
            xp = f_nl(xp) + sqrt(0.1) * randn(rng); xs[i] = xp
        end
        y = f_nl.(xs) .+ sqrt(0.5) .* randn(rng, n)
        init = @initialization begin
            μ(x) = NormalMeanVariance(0.0, 10.0)
        end
        return (iters -> infer(model = nl(), data = (y = y,), initialization = init, iterations = iters, options = (limit_stack_depth = 500,), free_energy = true)), 5
    elseif model == "hmm"
        n = 300
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
                    options = (limit_stack_depth = 500,), free_energy = true
                )
            ), 10
    end
    error("unknown model $model")
end

summarise(d) = d isa AbstractVector ? map(summarise, d) : (typeof(d), mean(d), d isa Union{Categorical, DirichletCollection} ? nothing : cov(d))

function main()
    run, iters = setup(MODEL)
    it1 = max(iters, 1)
    ttfx = @elapsed result = run(it1)
    fe = result.free_energy
    post = Dict(k => summarise(v isa AbstractVector && iters > 0 ? v : v) for (k, v) in pairs(result.posteriors))
    serialize(joinpath(PDIR, "$(VARIANT)_$(MODEL)_$(TAG).jls"), (post, fe))
    GC.gc()
    samples = [(GC.gc(); @timed run(it1)) for _ in 1:7]
    ts = map(s -> s.time, samples); bs = map(s -> s.bytes, samples)
    lines = String[]
    push!(lines, join((VARIANT, TAG, MODEL, "load", TLOAD, TLOAD, 0), '\t'))
    push!(lines, join((VARIANT, TAG, MODEL, "ttfx", ttfx, ttfx, 0), '\t'))
    push!(lines, join((VARIANT, TAG, MODEL, "infer(I=$it1)", minimum(ts), median(ts), minimum(bs)), '\t'))
    if iters > 0
        s2 = [(GC.gc(); @timed run(2iters)) for _ in 1:7]
        t2 = map(s -> s.time, s2); b2 = map(s -> s.bytes, s2)
        push!(lines, join((VARIANT, TAG, MODEL, "infer(I=$(2iters))", minimum(t2), median(t2), minimum(b2)), '\t'))
        push!(lines, join((VARIANT, TAG, MODEL, "per-iteration", (minimum(t2) - minimum(ts)) / iters, (median(t2) - median(ts)) / iters, (minimum(b2) - minimum(bs)) / iters), '\t'))
    end
    foreach(println, lines)
    return open(io -> foreach(l -> println(io, l), lines), OUT, "a")
end
main()
