module ReactiveMPModelsNonLinearDynamicsTest

using Test, InteractiveUtils
using Rocket, ReactiveMP, GraphPPL, Distributions
using BenchmarkTools, Random, Dates, StableRNGs, Flux

# Please use StableRNGs for random number generators

## Model definition
## -------------------------------------------- ##
sensor_location = 53
P = 5
sensor_var = 5
function f(z)
    (z - sensor_location)^2
end

@model function non_linear_dynamics(T, rng, n_iterations, n_samples, learning_rate)
    z = randomvar(T)
    x = randomvar(T)
    y = datavar(Float64, T)

    τ ~ GammaShapeRate(1.0, 1.0e-12)
    θ ~ GammaShapeRate(1.0, 1.0e-12)

    z[1] ~ NormalMeanPrecision(0, τ)
    x[1] ~ f(z[1]) where {meta = CVIApproximation(n_iterations, n_samples, rng, Descent(learning_rate), flux_update!)}
    y[1] ~ NormalMeanPrecision(x[1], θ)

    for t in 2:T
        z[t] ~ NormalMeanPrecision(z[t-1] + 1, τ)
        x[t] ~ f(z[t]) where {meta = CVIApproximation(n_iterations, n_samples, rng, Descent(learning_rate), flux_update!)}
        y[t] ~ NormalMeanPrecision(x[t], θ)
    end

    return z, x, y
end

constraints = @constraints begin
    q(z, x, τ, θ) = q(z)q(x)q(τ)q(θ)
end

## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function inference_cvi(transformed, rng)
    T = length(transformed)

    return inference(
        model = Model(non_linear_dynamics, T, rng, 1000, 2000, 0.1),
        data = (y = transformed,),
        iterations = 10,
        free_energy = false,
        returnvars = (z = KeepLast(),),
        constraints = constraints,
        initmessages = (z = NormalMeanVariance(0, P),),
        initmarginals = (
            z = NormalMeanVariance(0, P),
            τ = GammaShapeRate(1.0, 1.0e-12),
            θ = GammaShapeRate(1.0, 1.0e-12)
        )
    )
end

@testset "Non linear dynamics" begin
    @testset "Use case #1" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        seed = 123

        rng = MersenneTwister(seed)

        # For large `n` apply: smoothing(model_options(limit_stack_depth = 500), ...)
        T = 50

        sensor_location = 53

        hidden = collect(1:T)
        data = (hidden + rand(rng, NormalMeanVariance(0.0, sqrt(P)), T))
        transformed = (data .- sensor_location) .^ 2 + rand(rng, NormalMeanVariance(0.0, sensor_var), T)
        ## -------------------------------------------- ##
        ## Inference execution
        res = inference_cvi(transformed, rng)
        ## -------------------------------------------- ##
        ## Test inference results should be there
        ## -------------------------------------------- ##
        @test length(res.posteriors[:z]) === T
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        benchmark_output = joinpath(base_output, "non_linear_dynamics_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark = @benchmark inference_cvi($transformed, $rng)#
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end
end

end
