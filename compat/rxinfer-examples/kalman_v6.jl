# "Basic Examples/Kalman filtering and smoothing", its linear parts: the rotating state-space
# model (exact inference, free energy) and the smoothing model with missing data (VMP, Gamma
# prior on the observation precision). RxInfer 5.5.2 on ReactiveMP 6.5.0.
using RxInfer
include(joinpath(@__DIR__, "common.jl"))

const SIDE = "v6"
r = Recorder()

# --- rotate_ssm ------------------------------------------------------------------------------

seed = 1234
rng = MersenneTwister(seed)
θ = π / 35
A = [cos(θ) -sin(θ); sin(θ) cos(θ)]
B = diageye(2)
Q = 25.0 * diageye(2)
P = diageye(2)
n = 300
x, y = kalman_generate_data(rng, A, B, P, Q, n)
record!(r, "data:y", reduce(vcat, y))

@model function rotate_ssm(y, x0, A, B, P, Q)
    x_prior ~ x0
    x_prev = x_prior
    for i in 1:length(y)
        x[i] ~ MvNormalMeanCovariance(A * x_prev, P)
        y[i] ~ MvNormalMeanCovariance(B * x[i], Q)
        x_prev = x[i]
    end
end

x0 = MvNormalMeanCovariance(zeros(2), 100.0 * diageye(2))
result = infer(
    model = rotate_ssm(x0 = x0, A = A, B = B, P = P, Q = Q),
    data = (y = y,),
    free_energy = true
)
record!(r, "rotate_ssm:x", result.posteriors[:x])
record!(r, "rotate_ssm:free_energy", result.free_energy)

# --- smoothing with missing data ---------------------------------------------------------------

@model function smoothing(x0, y)
    τ ~ Gamma(shape = 0.001, scale = 0.001)
    x_prior ~ Normal(mean = mean(x0), var = var(x0))
    local x
    x_prev = x_prior
    for i in 1:length(y)
        x[i] ~ Normal(mean = x_prev, precision = 1.0)
        y[i] ~ Normal(mean = x[i], precision = τ)
        x_prev = x[i]
    end
end

real_signal, missing_data = missing_generate_data()
record!(r, "data:missing_y", coalesce.(missing_data, NaN))

constraints = @constraints begin
    q(x_prior, x, y, τ) = q(x_prior, x)q(τ)q(y)
end

x0_prior = NormalMeanVariance(0.0, 1000.0)
initm = @initialization begin
    q(τ) = Gamma(0.001, 0.001)
end

result = infer(
    model = smoothing(x0 = x0_prior),
    data = (y = missing_data,),
    constraints = constraints,
    initialization = initm,
    # The notebook keeps only `x`; `τ` is added to check the Gamma prior. RxInfer computes no
    # free energy for a model with missing observations.
    returnvars = (x = KeepLast(), τ = KeepEach()),
    iterations = 20
)
record!(r, "smoothing:x", result.posteriors[:x])
record!(r, "smoothing:τ", result.posteriors[:τ])

save_results(r, "kalman", SIDE)
