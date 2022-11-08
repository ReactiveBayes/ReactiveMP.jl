module ReactiveMPModelsSwitchTest

using Test, InteractiveUtils
using Rocket, ReactiveMP, GraphPPL, Distributions
using BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

# Please use StableRNGs for random number generators

## Model definition
## -------------------------------------------- ##
@model [addons = (AddonLogScale(),)] function beta_model1(n)
    y = datavar(Float64, n)

    θ ~ Beta(4.0, 8.0)

    for i in 1:n
        y[i] ~ Bernoulli(θ)
    end

    return y, θ
end
@model [addons = (AddonLogScale(),)] function beta_model2(n)
    y = datavar(Float64, n)

    θ ~ Beta(8.0, 4.0)

    for i in 1:n
        y[i] ~ Bernoulli(θ)
    end

    return y, θ
end
@model [addons = (AddonLogScale(),)] function beta_switch_model(n)
    y = datavar(Float64, n)

    selector ~ Bernoulli(0.7)

    in1 ~ Beta(4.0, 8.0)
    in2 ~ Beta(8.0, 4.0)

    θ ~ Switch(selector, (in1, in2))

    for i in 1:n
        y[i] ~ Bernoulli(θ)
    end

    return y, θ
end

@testset "Model switch" begin
    @testset "Check inference results" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        rng = MersenneTwister(42)
        n = 20
        θ_real = 0.75
        distribution = Bernoulli(θ_real)
        dataset = float.(rand(rng, Bernoulli(θ_real), n))

        ## -------------------------------------------- ##
        ## Inference execution
        result1 = inference(model = Model(beta_model1, length(dataset)), data = (y = dataset,), returnvars = (θ = KeepLast(),), free_energy = true)

        result2 = inference(model = Model(beta_model2, length(dataset)), data = (y = dataset,), returnvars = (θ = KeepLast(),), free_energy = true)

        resultswitch = inference(
            model = Model(beta_switch_model, length(dataset)), data = (y = dataset,), returnvars = (θ = KeepLast(), in1 = KeepLast(), in2 = KeepLast(), selector = KeepLast())
        )

        ## -------------------------------------------- ##
        ## Test inference results

        # check inference results
        @test getdata(result1.posteriors[:θ]) == getdata(resultswitch.posteriors[:in1])
        @test getdata(result2.posteriors[:θ]) == getdata(resultswitch.posteriors[:in2])
        @test getdata(resultswitch.posteriors[:in1]) == getdata(resultswitch.posteriors[:θ]).components[1]
        @test getdata(resultswitch.posteriors[:in2]) == getdata(resultswitch.posteriors[:θ]).components[2]
        @test getdata(resultswitch.posteriors[:selector]).p ≈ getdata(resultswitch.posteriors[:θ]).prior.p[1]

        # check free energies
        @test -result1.free_energy[1] ≈ getlogscale(result1.posteriors[:θ])
        @test -result2.free_energy[1] ≈ getlogscale(result2.posteriors[:θ])
        @test getlogscale(resultswitch.posteriors[:in1]) ≈ log(0.7) - result1.free_energy[1]
        @test getlogscale(resultswitch.posteriors[:in2]) ≈ log(0.3) - result2.free_energy[1]
        @test log(0.7 * exp(-result1.free_energy[1]) + 0.3 * exp(-result2.free_energy[1])) ≈ getlogscale(resultswitch.posteriors[:selector])
        @test log(0.7 * exp(-result1.free_energy[1]) + 0.3 * exp(-result2.free_energy[1])) ≈ getlogscale(resultswitch.posteriors[:θ])
        @test getlogscale(resultswitch.posteriors[:θ]) ≈ getlogscale(resultswitch.posteriors[:selector])

        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "switch_model_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "switch_model_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        rθ = range(0, 1, length = 1000)
        θestimated = resultswitch.posteriors[:θ]
        p = plot(title = "Inference results")

        plot!(rθ, (x) -> pdf(MixtureModel([Beta(4.0, 8.0), Beta(8.0, 4.0)], Categorical([0.5, 0.5])), x), fillalpha = 0.3, fillrange = 0, label = "P(θ)", c = 1)
        plot!(rθ, (x) -> pdf(getdata(θestimated), x), fillalpha = 0.3, fillrange = 0, label = "P(θ|y)", c = 3)
        vline!([θ_real], label = "Real θ")
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark = @benchmark 1 + 1#
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end
end

end
