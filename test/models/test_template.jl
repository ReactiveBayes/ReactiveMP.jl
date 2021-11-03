module ReactiveMPModelsLGSSMTest

using Test
using Rocket, ReactiveMP, GraphPPL, Distributions
using BenchmarkTools, Random, Plots, Dates, LinearAlgebra

## Model definition
## -------------------------------------------- ##
@model function template_model(n, x0, c_, P_)
    error("define model")
end
## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function template_inference(data, x0, c, P)
    error("define inference")
end

@testset "Model template" begin

    @testset "Use case template" begin 
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        _
        ## -------------------------------------------- ##
        ## Inference execution
        -
        ## -------------------------------------------- ##
        ## Test inference results
        _
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM") 
        plot_output      = joinpath(base_output, "template_model_plot_$(timestamp).png")
        benchmark_output = joinpath(base_output, "template_model_benchmark_$(timestamp).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks
        benchmark = @benchmark 1 + 1#
        open(benchmark_output, "w") do io
            show(io, MIME("text/plain"), benchmark)
        end
        ## -------------------------------------------- ##
    end

end

end