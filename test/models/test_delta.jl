module ReactiveMPModelsDeltaTest

using Test, InteractiveUtils
using Rocket, ReactiveMP, GraphPPL, Distributions
using BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

# Please use StableRNGs for random number generators

## Model definition
## -------------------------------------------- ##

function f₁(x)
    return sqrt.(x)
end

function f₁_inv(x)
    return x .^ 2
end

@model function delta_1input(meta)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    x ~ MvNormal(μ = ones(2), Λ = diageye(2))
    z ~ f₁(x) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

function f₂(x, θ)
    return x .+ θ
end

function f₂_x(θ, z)
    return z .- θ
end

function f₂_θ(x, z)
    return z .- x
end

@model function delta_2inputs(meta)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    θ ~ MvNormal(μ = ones(2), Λ = diageye(2))
    x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
    z ~ f₂(x, θ) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

function f₃(x, θ, ζ)
    return x .+ θ .+ ζ
end

@model function delta_3inputs(meta)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    θ ~ MvNormal(μ = ones(2), Λ = diageye(2))
    ζ ~ MvNormal(μ = 0.5ones(2), Λ = diageye(2))
    x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
    z ~ f₃(x, θ, ζ) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

function f₄(x, θ)
    return θ .* x
end

@model function delta_2input_1d2d(meta)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    θ ~ Normal(μ = 0.5, γ = 1.0)
    x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
    z ~ f₄(x, θ) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function inference_1input(data)
    res = []
    for meta in (ET(inverse = f₁_inv), UT(inverse = f₁_inv), ET(), UT())
        push!(
            res,
            inference(
                model = Model(delta_1input, meta),
                data = (y2 = data,),
                free_energy = true,
                free_energy_diagnostics = (BetheFreeEnergyCheckNaNs(), BetheFreeEnergyCheckInfs())
            )
        )
    end
    res
end

function inference_2inputs(data)
    res = []
    for meta in (ET(inverse = (f₂_x, f₂_θ)), UT(inverse = (f₂_x, f₂_θ)), ET(), UT())
        push!(
            res,
            inference(
                model = Model(delta_2inputs, meta),
                data = (y2 = data,),
                free_energy = true,
                free_energy_diagnostics = (BetheFreeEnergyCheckNaNs(), BetheFreeEnergyCheckInfs())
            )
        )
    end
    res
end

function inference_3inputs(data)
    res = []
    for meta in (ET(), UT())
        push!(
            res,
            inference(
                model = Model(delta_3inputs, meta),
                data = (y2 = data,),
                free_energy = true,
                free_energy_diagnostics = (BetheFreeEnergyCheckNaNs(), BetheFreeEnergyCheckInfs())
            )
        )
    end
    res
end

function inference_2input_1d2d(data)
    res = []
    for meta in (ET(), UT())
        push!(
            res,
            inference(
                model = Model(delta_2input_1d2d, meta),
                data = (y2 = data,),
                free_energy = true,
                free_energy_diagnostics = (BetheFreeEnergyCheckNaNs(), BetheFreeEnergyCheckInfs())
            )
        )
    end
    res
end

@testset "Delta models" begin
    @testset "Extended, Unscented transforms" begin
        ## -------------------------------------------- ##
        ## Data creation
        data = 4.0
        ## -------------------------------------------- ##
        ## Inference execution
        result₁ = inference_1input(data)
        result₂ = inference_2inputs(data)
        result₃ = inference_3inputs(data)
        result₄ = inference_2input_1d2d(data)
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "template_model_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "template_model_benchmark_$(timestamp)_v$(VERSION).txt")
    end
end

end
