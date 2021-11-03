module ReactiveMPModelsLGSSMTest

using Test
using Rocket, ReactiveMP, GraphPPL, Distributions
using BenchmarkTools, Random, Plots, Dates

## Model definition
## -------------------------------------------- ##
@model function univariate_lgssm_model(n, x0, c_, P_)

    x_prior ~ NormalMeanVariance(mean(x0), cov(x0)) 

    x = randomvar(n)
    c = constvar(c_)
    P = constvar(P_)
    y = datavar(Float64, n)
    
    x_prev = x_prior
    
    for i in 1:n
        x[i] ~ x_prev + c
        y[i] ~ NormalMeanVariance(x[i], P)
        x_prev = x[i]
    end

    return x, y
end
## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function univariate_lgssm_inference(data, x0, c, P)
    n = length(data)
    
    model, (x, y) = univariate_lgssm_model(n, x0, c, P);

    x_buffer = buffer(Marginal, n)
    fe       = 0.0
    
    x_sub = subscribe!(getmarginals(x), x_buffer)
    f_sub = subscribe!(score(Float64, BetheFreeEnergy(), model), (v) -> fe = v)
    
    update!(y, data)
    
    unsubscribe!((x_sub, f_sub))
    
    return x_buffer, fe
end

@testset "Linear Gaussian State Space Model" begin

    @testset "Univariate" begin 

        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        rng = MersenneTwister(123)
        P   = 100.0
        n   = 500
        hidden   = collect(1:n)
        data     = hidden + rand(rng, Normal(0.0, sqrt(P)), n);
        x0_prior = NormalMeanVariance(0.0, 10000.0)
        ## -------------------------------------------- ##
        ## Inference execution
        x_estimated, fe = univariate_lgssm_inference(data, x0_prior, 1.0, P);
        ## -------------------------------------------- ##
        ## Test inference results
        @test length(x_estimated) === n
        @test all((hidden .- 3 .* std.(x_estimated)) .< mean.(x_estimated) .< (hidden .+ 3 .* std.(x_estimated)))
        @test all(var.(x_estimated) .> 0.0)
        @test fe ≈ 1896.15674252578
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd:mm:yyyy-HH:MM") 
        plot_output      = joinpath(base_output, "lgssm_univariate_plot_$(timestamp).png")
        benchmark_output = joinpath(base_output, "lgssm_univariate_benchmark_$(timestamp).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        subrange = 200:215
        m = mean.(x_estimated)[subrange]
        s = std.(x_estimated)[subrange]
        p = plot()
        p = plot!(subrange, m, ribbon = s, label = "Estimated signal")
        p = plot!(subrange, hidden[subrange], label = "Hidden signal")
        p = scatter!(subrange, data[subrange], label = "Observations")
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks
        benchmark = @benchmark univariate_lgssm_inference($data, $x0_prior, 1.0, $P)
        open(benchmark_output, "w") do io
            show(io, MIME("text/plain"), benchmark)
        end
        ## -------------------------------------------- ##
    end

end

end