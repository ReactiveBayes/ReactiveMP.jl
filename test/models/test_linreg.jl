module ReactiveMPModelsLinearRegressionTest

using Test, InteractiveUtils
using Rocket, ReactiveMP, GraphPPL, Distributions
using BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

# Please use StableRNGs for random number generators

## Model definition
## -------------------------------------------- ##
@model function linear_regression(n)
    a ~ NormalMeanVariance(0.0, 1.0)
    b ~ NormalMeanVariance(0.0, 1.0)
    
    x = datavar(Float64, n)
    y = datavar(Float64, n)
    
    for i in 1:n
        y[i] ~ NormalMeanVariance(x[i] * b + a, 1.0)
    end
    
    return a, b, x, y
end
## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function inference(xdata, ydata)
    @assert length(xdata) == length(ydata)
    
    n = length(xdata)
    
    model, (a, b, x, y) = linear_regression(n)
    
    as = storage(Marginal)
    bs = storage(Marginal)
    fe = ScoreActor(Float64)
    
    asub = subscribe!(getmarginal(a), as)
    bsub = subscribe!(getmarginal(b), bs)
    fsub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)
    
    setmessage!(b, NormalMeanVariance(0.0, 100.0))
    
    for i in 1:25
        update!(x, xdata)
        update!(y, ydata)
    end

    unsubscribe!(asub)
    unsubscribe!(bsub)
    unsubscribe!(fsub)
    
    return getvalues(as), getvalues(bs), getvalues(fe)
end

@testset "Linear regression" begin

    @testset "Use case #1" begin 
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        reala = 10.0
        realb = -10.0

        N = 100

        rng = StableRNG(1234)

        xdata = collect(1:N) .+ 1 * randn(rng, N)
        ydata = reala .+ realb .* xdata;
        ## -------------------------------------------- ##
        ## Inference execution
        ares, bres, fres = inference(xdata, ydata);
        ## -------------------------------------------- ##
        ## Test inference results
        @test isapprox(mean(ares), reala, atol = 5)
        @test isapprox(mean(bres), realb, atol = 0.1)
        @test fres[end] < fres[2] # Loopy belief propagation has no guaranties though
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM") 
        benchmark_output = joinpath(base_output, "linear_regression_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output benchmarks
        benchmark = @benchmark inference($xdata, $ydata);#
        open(benchmark_output, "w") do io
            show(io, MIME("text/plain"), benchmark)
            versioninfo(io)
        end
        ## -------------------------------------------- ##
    end

end

end