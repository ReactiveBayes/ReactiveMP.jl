module ReactiveMPModelsAutoregressiveTest

using Test, InteractiveUtils
using Rocket, ReactiveMP, GraphPPL, Distributions
using BenchmarkTools, Random, Plots, Dates, LinearAlgebra

## Model definition
## -------------------------------------------- ##
@model [ default_factorisation = MeanField() ] function lar_model(T::Type{ Multivariate }, n, order, c, stype, τ)

    # Parameter priors
    γ  ~ GammaShapeRate(1.0, 1.0)
    θ  ~ MvNormalMeanPrecision(zeros(order), diageye(order))

    # We create a sequence of random variables for hidden states
    x = randomvar(n)
    # As well a sequence of observartions
    y = datavar(Float64, n)

    ct = constvar(c)
    # We assume observation noise to be known
    cτ = constvar(τ)

    # Prior for first state
    x0 ~ MvNormalMeanPrecision(zeros(order), diageye(order))

    x_prev = x0

    # AR process requires extra meta information
    meta = ARMeta(Multivariate, order, stype)

    for i in 1:n
        # Autoregressive node uses structured factorisation assumption between states
        x[i] ~ AR(x_prev, θ, γ) where { q = q(y, x)q(γ)q(θ), meta = meta }
        y[i] ~ NormalMeanPrecision(dot(ct, x[i]), cτ)
        x_prev = x[i]
    end

    return x, y, θ, γ
end
@model [ default_factorisation = MeanField() ] function lar_model(T::Type{Univariate}, n, order, c, stype, τ)

    # Parameter priors
    γ  ~ GammaShapeRate(1.0, 1.0)
    θ  ~ NormalMeanPrecision(0.0, 1.0)

    # We create a sequence of random variables for hidden states
    x = randomvar(n)
    # As well a sequence of observartions
    y = datavar(Float64, n)

    ct = constvar(c)
    # We assume observation noise to be known
    cτ = constvar(τ)

    # Prior for first state
    x0 ~ NormalMeanPrecision(0.0, 1.0)

    x_prev = x0

    # AR process requires extra meta information
    meta = ARMeta(Univariate, order, stype)

    for i in 1:n
        x[i] ~ AR(x_prev, θ, γ) where { q = q(y, x)q(γ)q(θ), meta = meta }
        y[i] ~ NormalMeanPrecision(ct * x[i], cτ)
        x_prev = x[i]
    end

    return x, y, θ, γ
end
## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function init_marginals!(::Type{ Multivariate }, order, γ, θ)
    setmarginal!(γ, GammaShapeRate(1.0, 1.0))
    setmarginal!(θ, MvNormalMeanPrecision(zeros(order), diageye(order)))
end

function init_marginals!(::Type{ Univariate }, order, γ, θ)
    setmarginal!(γ, GammaShapeRate(1.0, 1.0))
    setmarginal!(θ, NormalMeanPrecision(0.0, 1.0))
end

function inference(data, order, artype, stype, niter, τ)

    # We build a full graph based on nber of observatios
    n = length(data)

    # Depending on the order of AR process `c` is
    # either a nber or a vector
    c = ReactiveMP.ar_unit(artype, order)

    # Note that to run inference for huge model it might be necessary to pass extra
    # options = (limit_stack_depth = 100,) to limit stack depth during recursive inference procedure
    model, (x, y, θ, γ) = lar_model(artype, n, order, c, stype, τ)

    # We are going to keep `γ` and `θ` estimates for all VMP iterations
    # But `buffer` only last posterior estimates for a sequence of hidden states `x`
    # We also will keep Bethe Free Energy in `fe`
    γ_buffer = keep(Marginal)
    θ_buffer = keep(Marginal)
    x_buffer = buffer(Marginal, n)
    fe       = keep(Float64)

    γsub  = subscribe!(getmarginal(γ), γ_buffer)
    θsub  = subscribe!(getmarginal(θ), θ_buffer)
    xsub  = subscribe!(getmarginals(x), x_buffer)
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    init_marginals!(artype, order, γ, θ)

    # We update data several times to perform several VMP iterations
    for i in 1:niter
        update!(y, data)
    end

    # It is important to unsubscribe from running observables
    unsubscribe!((γsub, θsub, xsub, fesub))

    return γ_buffer, θ_buffer, x_buffer, fe
end

@testset "Model template" begin

    @testset "Use case template" begin 
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        # The following coefficients correspond to stable poles
        coefs_ar_5 = [ 0.10699399235785655, -0.5237303489793305, 0.3068897071844715, -0.17232255282458891, 0.13323964347539288 ]

        function generate_ar_data(rng, n, θ, γ, τ)
            order        = length(θ)
            states       = Vector{Vector{Float64}}(undef, n + 3order)
            observations = Vector{Float64}(undef, n + 3order)
        
            γ_std = sqrt(inv(γ))
            τ_std = sqrt(inv(γ))
        
            states[1] = randn(rng, order)
        
            for i in 2:(n + 3order)
                states[i]       = vcat(rand(rng, Normal(dot(θ, states[i - 1]), γ_std)), states[i-1][1:end-1])
                observations[i] = rand(rng, Normal(states[i][1], τ_std))
            end
        
            return states[1+3order:end], observations[1+3order:end]
        end
        # Seed for reproducibility
        rng  = MersenneTwister(123)
        # Number of observations in synthetic dataset
        n = 500
        # AR process parameters
        real_γ = 5.0
        real_τ = 5.0
        real_θ = coefs_ar_5
        states, observations = generate_ar_data(rng, n, real_θ, real_γ, real_τ)
        ## -------------------------------------------- ##
        ## Inference execution

        # AR order 1
        for i in 2:5
            γ, θ, xs, fe = inference(observations, i, Univariate, ARsafe(), 15, real_τ)
            @test length(xs) === n
            @test length(γ)  === 15
            @test length(θ)  === 15
            @test length(fe) === 15
        end

        γ, θ, xs, fe = inference(observations, 1, Univariate, ARsafe(), 15, real_τ)
        @test length(xs) === n
        @test length(γ)  === 15
        @test length(θ)  === 15
        @test length(fe) === 15 && last(fe) ≈ 535.3776616955
        @test all(diff(fe) .< 0)

        for i in 1:4
            γ, θ, xs, fe = inference(observations, i, Multivariate, ARsafe(), 15, real_τ)
            @test length(xs) === n
            @test length(γ)  === 15
            @test length(θ)  === 15
            @test length(fe) === 15
        end

        # AR order 5
        γ, θ, xs, fe = inference(observations, length(real_θ), Multivariate, ARsafe(), 15, real_τ)
        @test length(xs) === n
        @test length(γ)  === 15
        @test length(θ)  === 15
        @test length(fe) === 15 && last(fe) ≈ 524.0689496230
        @test all(diff(fe) .< 0)
        @test (mean(last(γ)) - 3.0std(last(γ)) < real_γ < mean(last(γ)) + 3.0std(last(γ)))

        sreal_θ = sort(real_θ)
        sθ      = sort(ReactiveMP.getvalues(θ), by = mean)

        foreach(zip(sreal_θ, sθ)) do (real, estimated)
            @test mean(estimated) - 3std(estimated) < real < mean(estimated) + 3std(estimated) 
        end

        ## -------------------------------------------- ##
        ## Test inference results
        _
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM") 
        plot_output      = joinpath(base_output, "autoregressive_model_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "autoregressive_model_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        p1 = plot(first.(states), label="Hidden state")
        p1 = scatter!(p1, observations, label="Observations")
        p1 = plot!(p1, first.(mean.(xs)), ribbon = sqrt.(first.(var.(xs))), label="Inferred states", legend = :bottomright)

        p2 = plot(mean.(γ), ribbon = std.(γ), label = "Inferred transition precision", legend = :bottomright)
        p2 = plot!([ real_γ ], seriestype = :hline, label = "Real transition precision")

        p3 = plot(getvalues(fe), label = "Bethe Free Energy")

        p = plot(p1, p2, p3, layout = @layout([ a; b c ]))
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks
        benchmark = @benchmark inference($observations, length($real_θ), Multivariate, ARsafe(), 15, $real_τ)#
        open(benchmark_output, "w") do io
            show(io, MIME("text/plain"), benchmark)
            versioninfo(io)
        end
        ## -------------------------------------------- ##
    end

end

end