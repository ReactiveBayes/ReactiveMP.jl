# Model creation alone, split by stage, across model sizes.
#   julia --project=<variant>/env bench_creation.jl <variant> <tag> <outfile> <model> [sizes...]
# Stages, each a separate `create_model` of the same conditioned generator:
#   graph0  GraphPPL's graph only: RxInfer's backend, no plugins
#   graph   + the constraints, meta (algorithm) and initialization plugins
#   full    + RxInfer's ReactiveMP plugin and the free-energy plugin (Float64): what
#           `batch_inference` builds, ReactiveMP variables, nodes and their activation included
# `first_full` is the first `full` creation at the smallest size (compile time included).
# One TSV line per (model, n, stage): variant, tag, model, n, stage, min s, median s, min bytes.

using RxInfer, StableRNGs, LinearAlgebra, Statistics
import GraphPPL
using DeltaMessagePassingRules, DiscreteTransitionMessagePassingRules

const VARIANT = ARGS[1]
const TAG = ARGS[2]
const OUT = ARGS[3]
const MODEL = ARGS[4]
const SIZES = length(ARGS) > 4 ? parse.(Int, ARGS[5:end]) : [100, 1000, 10000]
const STAGES = split(get(ENV, "CREATION_STAGES", "graph0,graph,full"), ",")

@model function ssm1(y, P)
    x_prior ~ Normal(mean = 0.0, variance = 10000.0)
    x_prev = x_prior
    for i in eachindex(y)
        x[i] ~ Normal(mean = x_prev, variance = 1.0)
        y[i] ~ Normal(mean = x[i], variance = P)
        x_prev = x[i]
    end
end

@model function iid(y)
    μ ~ Normal(mean = 0.0, variance = 100.0)
    τ ~ Gamma(shape = 1.0, rate = 1.0)
    for i in eachindex(y)
        y[i] ~ Normal(mean = μ, precision = τ)
    end
end

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

f_nl(x) = x + 0.1 * sin(x)
# the algorithm of every Delta node from an `@algorithm` block: the meta plugin's `apply_meta!`
@model function nl(y)
    x_prior ~ Normal(mean = 0.0, variance = 10.0)
    x_prev = x_prior
    for i in eachindex(y)
        x[i] ~ Normal(mean = x_prev, variance = 0.1)
        z[i] := f_nl(x[i])
        y[i] ~ Normal(mean = z[i], variance = 0.5)
        x_prev = x[i]
    end
end

# a Kalman step as a submodel: one Context per time step
@model function kstep(x_next, x_prev, y)
    x_next ~ Normal(mean = x_prev, variance = 1.0)
    y ~ Normal(mean = x_next, variance = 10.0)
end
@model function ssmsub(y)
    x_prior ~ Normal(mean = 0.0, variance = 10000.0)
    x_prev = x_prior
    for i in eachindex(y)
        x[i] ~ kstep(x_prev = x_prev, y = y[i])
        x_prev = x[i]
    end
end

# a matrix variable under a factorisation constraint
@model function mat(y, m)
    τ ~ Gamma(shape = 1.0, rate = 1.0)
    for i in 1:m, j in 1:m
        x[i, j] ~ Normal(mean = 0.0, precision = τ)
        y[i, j] ~ Normal(mean = x[i, j], variance = 1.0)
    end
end

function onehot(rng, p)
    s = zeros(length(p)); s[rand(rng, Categorical(p))] = 1.0
    return s
end

# (generator, data, constraints, meta, initialization) for a size n
function setup(model, n)
    rng = StableRNG(42)
    if model == "ssm1"
        return ssm1(P = 10.0), (y = randn(rng, n),), nothing, nothing, nothing
    elseif model == "iid"
        init = @initialization begin
            q(τ) = GammaShapeRate(1.0, 1.0)
        end
        return iid(), (y = randn(rng, n),), MeanField(), nothing, init
    elseif model == "hmm"
        x = [onehot(rng, [0.3, 0.3, 0.4]) for _ in 1:n]
        cons = @constraints begin
            q(s, s_0, A, B) = q(s, s_0)q(A)q(B)
        end
        init = @initialization begin
            q(A) = vague(DirichletCollection, (3, 3))
            q(B) = vague(DirichletCollection, (3, 3))
            q(s) = vague(Categorical, 3)
        end
        return hmm(), (x = x,), cons, nothing, init
    elseif model == "nl"
        alg = @algorithm begin
            f_nl() -> DeltaApproximation(method = Linearization())
        end
        init = @initialization begin
            μ(x) = NormalMeanVariance(0.0, 10.0)
        end
        return nl(), (y = randn(rng, n),), nothing, alg, init
    elseif model == "ssmsub"
        return ssmsub(), (y = randn(rng, n),), nothing, nothing, nothing
    elseif model == "mat"
        m = round(Int, sqrt(n))
        cons = @constraints begin
            q(x, τ) = q(x)q(τ)
        end
        init = @initialization begin
            q(τ) = GammaShapeRate(1.0, 1.0)
        end
        return mat(m = m), (y = randn(rng, m, m),), cons, nothing, init
    end
    error("unknown model $model")
end

function plugins_for(stage, constraints, meta, initialization)
    stage == "graph0" && return GraphPPL.PluginsCollection()
    base = GraphPPL.PluginsCollection(
        GraphPPL.VariationalConstraintsPlugin(constraints),
        GraphPPL.MetaPlugin(meta),
        RxInfer.InitializationPlugin(initialization),
    )
    stage == "graph" && return base
    options = RxInfer.setwarn(convert(RxInfer.ReactiveMPInferenceOptions, (limit_stack_depth = 500,)), true)
    return base + RxInfer.ReactiveMPInferencePlugin(options) + RxInfer.ReactiveMPFreeEnergyPlugin(RxInfer.BetheFreeEnergy(Float64))
end

function create(stage, generator, data, constraints, meta, initialization)
    plugins = plugins_for(stage, constraints, meta, initialization)
    model = GraphPPL.with_backend(GraphPPL.with_plugins(generator, plugins), RxInfer.ReactiveMPGraphPPLBackend(RxInfer.Static.static(false)))
    return RxInfer.create_model(model | data)
end

emit(line) = (println(line); open(io -> println(io, line), OUT, "a"))

function main()
    first_n = minimum(SIZES)
    gen, data, cons, meta, init = setup(MODEL, first_n)
    t = @elapsed create("full", gen, data, cons, meta, init)
    emit(join((VARIANT, TAG, MODEL, first_n, "first_full", t, t, 0), '\t'))
    for n in SIZES, stage in STAGES
        gen, data, cons, meta, init = setup(MODEL, n)
        create(stage, gen, data, cons, meta, init)
        k = n >= 100_000 ? 2 : n >= 10_000 ? 4 : 8
        samples = [(GC.gc(); @timed create(stage, gen, data, cons, meta, init)) for _ in 1:k]
        ts = map(s -> s.time, samples)
        emit(join((VARIANT, TAG, MODEL, n, stage, minimum(ts), median(ts), minimum(s -> s.bytes, samples)), '\t'))
    end
    return
end
isdefined(Main, :NO_MAIN) || main()
