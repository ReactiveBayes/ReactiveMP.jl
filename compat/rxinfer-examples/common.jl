# Shared by every `<name>_v6.jl` and `<name>_v7.jl`: the data generators of the notebooks, copied
# verbatim so that both sides see identical data, and the recorder that writes results.
#
# Loaded after `using RxInfer` (either version), so the distribution types resolve to the same
# ExponentialFamily/Distributions versions on both sides.

using Serialization, LinearAlgebra, Random, StableRNGs, DelimitedFiles

# ExponentialFamily is loaded by RxInfer on both sides but is a direct dependency of neither
# example environment; reach it through the loaded modules.
const ExponentialFamily = Base.loaded_modules[Base.PkgId(Base.UUID("62312e5e-252a-4322-ace9-a5f4bf9b357b"), "ExponentialFamily")]

const RESULTS_DIR = joinpath(@__DIR__, "results")

# Which engine this process runs, so that a v6 run that silently loaded the local v7 checkout
# (or the reverse) shows in the output.
println("RxInfer ", pkgversion(RxInfer), " on ReactiveMP ", pkgversion(RxInfer.ReactiveMP), " (", pathof(RxInfer.ReactiveMP), ")")

# ---------------------------------------------------------------------------------------------
# Recording. A result is a `Dict{String, Vector{Float64}}`: one entry per posterior (per index
# and per iteration), holding a summary of its parameters, plus `Dict{String, String}` of the
# posterior's type, for information only (compare.jl reports a type change but compares values).

struct Recorder
    values::Dict{String, Vector{Float64}}
    types::Dict{String, String}
end
Recorder() = Recorder(Dict{String, Vector{Float64}}(), Dict{String, String}())

flat(x::Real) = [Float64(x)]
flat(x::AbstractArray) = Float64.(vec(collect(x)))

# The parameters each family the examples produce is compared on: moments for the Gaussians
# (whatever their parameterisation), the canonical parameters otherwise.
summarise(d::UnivariateNormalDistributionsFamily) = vcat(flat(mean(d)), flat(var(d)))
summarise(d::MultivariateNormalDistributionsFamily) = vcat(flat(mean(d)), flat(cov(d)))
summarise(d::GammaDistributionsFamily) = vcat(flat(shape(d)), flat(rate(d)))
summarise(d::Beta) = flat(collect(params(d)))
summarise(d::Categorical) = flat(probvec(d))
summarise(d::Bernoulli) = flat(succprob(d))
summarise(d::Dirichlet) = flat(d.alpha)
summarise(d::DirichletCollection) = vcat(flat(mean(d)), flat(d.α))
summarise(d::Union{Wishart, ExponentialFamily.WishartFast}) = vcat(flat(mean(d)), flat(var(d)))
summarise(d::PointMass) = flat(mean(d))
function summarise(d)
    # Anything else (a SampleList, a mixture, …): its first two moments.
    return vcat(flat(mean(d)), flat(d isa UnivariateDistribution ? var(d) : cov(d)))
end

record!(r::Recorder, name::String, x::Real) = (r.values[name] = flat(x); r.types[name] = string(typeof(x)); r)
function record!(r::Recorder, name::String, xs::AbstractArray)
    if eltype(xs) <: Real
        r.values[name] = flat(xs)
        r.types[name] = string(typeof(xs))
    else
        for i in eachindex(xs)
            record!(r, string(name, "[", i, "]"), xs[i])
        end
    end
    return r
end
function record!(r::Recorder, name::String, d)
    r.values[name] = summarise(d)
    r.types[name] = string(nameof(typeof(d)))
    return r
end

function save_results(r::Recorder, example::String, side::String)
    mkpath(RESULTS_DIR)
    path = joinpath(RESULTS_DIR, "$(example)_$(side).jls")
    serialize(path, (values = r.values, types = r.types))
    println("wrote $(length(r.values)) entries to $path")
    return path
end

# ---------------------------------------------------------------------------------------------
# Kalman filtering and smoothing, part 1: the rotating state-space model.

function kalman_generate_data(rng, A, B, P, Q, n)
    x_prev = [10.0, -10.0]
    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        x[i] = rand(rng, MvNormalMeanCovariance(A * x_prev, P))
        y[i] = rand(rng, MvNormalMeanCovariance(B * x[i], Q))
        x_prev = x[i]
    end
    return x, y
end

# Kalman filtering and smoothing, parts 2–4: the identification problems (`+`, `smooth_min`).

function identification_generate_data(f, n; seed = 123, x_i_min = -20.0, w_i_min = 20.0, noise = 20.0, real_x_τ = 0.1, real_w_τ = 1.0)
    rng = StableRNG(seed)
    real_x = Vector{Float64}(undef, n)
    real_w = Vector{Float64}(undef, n)
    real_y = Vector{Float64}(undef, n)
    for i in 1:n
        real_x[i] = rand(rng, Normal(x_i_min, sqrt(1.0 / real_x_τ)))
        real_w[i] = rand(rng, Normal(w_i_min, sqrt(1.0 / real_w_τ)))
        real_y[i] = rand(rng, Normal(f(real_x[i], real_w[i]), sqrt(noise)))
        x_i_min = real_x[i]
        w_i_min = real_w[i]
    end
    return real_x, real_w, real_y
end

# Smoothed version of `min` without zero-ed derivatives
function smooth_min(x, y)
    if x < y
        return x + 1.0e-4 * y
    else
        return y + 1.0e-4 * x
    end
end

# Kalman filtering and smoothing, part 5: smoothing with missing data. The notebook draws the
# noise from the global generator, `rand(Normal(0.0, 1 / sqrt(τ)), n)`, which is not
# reproducible; here it is drawn from `StableRNG(42)`, the only departure from the notebook.
function missing_generate_data(; τ = 1.0, n = 250, rng = StableRNG(42))
    real_signal = map(e -> sin(0.05 * e), collect(1:n))
    noisy_data = real_signal + rand(rng, Normal(0.0, 1 / sqrt(τ)), n)
    missing_indices = 100:125
    missing_data = similar(noisy_data, Union{Float64, Missing})
    copyto!(missing_data, noisy_data)
    for index in missing_indices
        missing_data[index] = missing
    end
    return real_signal, missing_data
end

# ---------------------------------------------------------------------------------------------
# Hidden Markov Model.

function hmm_rand_vec(rng, distribution::Categorical)
    k = ncategories(distribution)
    s = zeros(k)
    drawn_category = rand(rng, distribution)
    s[drawn_category] = 1.0
    return s
end

function hmm_generate_data(n_samples; seed = 42)
    rng = MersenneTwister(seed)
    state_transition_matrix = [
        0.9 0.05 0.0;
        0.1 0.9 0.1;
        0.0 0.05 0.9
    ]
    observation_distribution_matrix = [
        0.9 0.05 0.05;
        0.05 0.9 0.05;
        0.05 0.05 0.9
    ]
    s_initial = [1.0, 0.0, 0.0]
    states = Vector{Vector{Float64}}(undef, n_samples)
    observations = Vector{Vector{Float64}}(undef, n_samples)
    s_prev = s_initial
    for t in 1:n_samples
        s_probvec = state_transition_matrix * s_prev
        states[t] = hmm_rand_vec(rng, Categorical(s_probvec ./ sum(s_probvec)))
        obs_probvec = observation_distribution_matrix * states[t]
        observations[t] = hmm_rand_vec(rng, Categorical(obs_probvec ./ sum(obs_probvec)))
        s_prev = states[t]
    end
    return observations, states
end

# ---------------------------------------------------------------------------------------------
# Gaussian Mixture.

function gmm_generate_univariate_data(nr_samples; rng = MersenneTwister(123))
    class = [1 / 3, 2 / 3]
    mean1, mean2 = -10, 10
    precision = 1.777
    z = rand(rng, Categorical(class), nr_samples)
    y = zeros(nr_samples)
    for k in 1:nr_samples
        y[k] = rand(rng, Normal(z[k] == 1 ? mean1 : mean2, 1 / sqrt(precision)))
    end
    return y
end

function gmm_generate_multivariate_data(nr_samples; rng = MersenneTwister(123))
    L = 50.0
    nr_mixtures = 6
    probvec = normalize!(ones(nr_mixtures), 1)
    switch = Categorical(probvec)
    gaussians = map(1:nr_mixtures) do index
        angle = 2π / nr_mixtures * (index - 1)
        basis_v = L * [1.0, 0.0]
        R = [cos(angle) -sin(angle); sin(angle) cos(angle)]
        mean = R * basis_v
        covariance = Matrix(Hermitian(R * [10.0 0.0; 0.0 20.0] * transpose(R)))
        return MvNormal(mean, covariance)
    end
    z = rand(rng, switch, nr_samples)
    y = Vector{Vector{Float64}}(undef, nr_samples)
    for n in 1:nr_samples
        y[n] = rand(rng, gaussians[z[n]])
    end
    return y
end

# ---------------------------------------------------------------------------------------------
# Autoregressive Models.

const coefs_ar_5 = [0.10699399235785655, -0.5237303489793305, 0.3068897071844715, -0.17232255282458891, 0.13323964347539288]

function ar_generate_synthetic_dataset(; n, θ, γ = 1.0, τ = 1.0, rng = StableRNG(42), states1 = randn(rng, length(θ)))
    order = length(θ)
    states = Vector{Vector{Float64}}(undef, n + 3order)
    observations = Vector{Float64}(undef, n + 3order)
    states[1] = states1
    observations[1] = rand(rng, NormalMeanPrecision(states[1][1], γ))
    for i in 2:(n + 3order)
        previous_state = states[i - 1]
        transition = dot(θ, previous_state)
        next_x = rand(rng, NormalMeanPrecision(transition, τ))
        states[i] = vcat(next_x, previous_state[1:(end - 1)])
        observations[i] = rand(rng, NormalMeanPrecision(next_x, γ))
    end
    return states[(1 + 3order):end], observations[(1 + 3order):end]
end

function ar_generate_sinusoidal_coefficients(; f)
    a1 = 2cos(2pi * f)
    a2 = -1
    return [a1, a2]
end

# The notebook reads `aal_stock.csv` with CSV.jl and DataFrames.jl and keeps the non-missing
# entries of the fifth column, "close"; `readdlm` reads the same file (it has no missing entries).
const AAL_STOCK_CSV = joinpath(@__DIR__, "..", "..", "..", "RxInferExamples.jl", "examples", "Problem Specific", "Autoregressive Models", "aal_stock.csv")

function ar_stock_data()
    table, header = readdlm(AAL_STOCK_CSV, ','; header = true)
    @assert header[5] == "close"
    return Float64.(table[:, 5])
end

function ar_shift(dim)
    S = Matrix{Float64}(I, dim, dim)
    for i in dim:-1:2
        S[i, :] = S[i - 1, :]
    end
    S[1, :] = zeros(dim)
    return S
end
