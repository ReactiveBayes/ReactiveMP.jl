module ReactiveMPRenderCVITest

using Test
using ReactiveMP
using Random
using StableRNGs
using Distributions
using Zygote
using Flux

import ReactiveMP: naturalparams, NaturalParameters

struct EmptyOptimizer end

function ReactiveMP.cvi_update!(::EmptyOptimizer, λ, ∇)
    return vec(λ)
end

mutable struct CountingOptimizer
    num_its::Int
end

function ReactiveMP.cvi_update!(opt::CountingOptimizer, λ, ∇)
    opt.num_its += 1
    return vec(λ)
end

@testset "cvi:prod" begin
    @testset "empty optimizer" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            opt = EmptyOptimizer()
            meta = CVI(1, 100, opt)
            λ = prod(meta, (z) -> logpdf(m_out, z), m_in)
            @test all(mean_var(convert(Distribution, λ)) .== mean_var(m_in))
        end
    end

    @testset "counting optimizer" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            opt = CountingOptimizer(0)
            meta = CVI(1, 100, opt)
            λ = prod(meta, (z) -> logpdf(m_out, z), m_in)
            @test all(mean_var(convert(Distribution, λ)) .== mean_var(m_in))
            @test opt.num_its === 100
        end
    end

    @testset "counting lambda optimizer" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            η = naturalparams(m_in)
            logp_nc = (z) -> logpdf(m_out, z)
            num_its = 0
            opt = (λ, ∇) -> begin
                num_its += 1
                return vec(λ)
            end
            meta = CVI(1, 100, opt)
            λ = prod(meta, (z) -> logpdf(m_out, z), m_in)
            @test all(mean_var(convert(Distribution, λ)) .== mean_var(m_in))
            @test num_its === 100
        end
    end

    @testset "counting lambda optimizer (Zygote Grad)" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            η = naturalparams(m_in)
            logp_nc = (z) -> logpdf(m_out, z)
            num_its = 0
            opt = (λ, ∇) -> begin
                num_its += 1
                return vec(λ)
            end
            meta = CVI(1, 100, opt, ZygoteGrad())
            λ = prod(meta, (z) -> logpdf(m_out, z), m_in)
            @test all(mean_var(convert(Distribution, λ)) .== mean_var(m_in))
            @test num_its === 100
        end
    end

    @testset "Normal x Normal (Specific prod)" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Descent(0.01)
        meta = CVI(rng, 1, 1000, optimizer, ForwardDiffGrad(), false, false)

        for i in 1:10
            m_out, m_in = NormalMeanVariance(i, 1), NormalMeanVariance(0, 1)
            λ = prod(meta, (z) -> logpdf(m_out, z), m_in)
            @test isapprox(convert(Distribution, λ), NormalWeightedMeanPrecision(i, 2), atol = 0.1)
        end
    end

    @testset "Normal x Normal (Generic prod)" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Descent(0.001)
        meta = CVI(rng, 1, 5000, optimizer, ForwardDiffGrad(), false, false)

        for i in 1:3
            m_out, m_in = NormalMeanVariance(i, 1), NormalMeanVariance(0, 1)
            λ = Base.@invoke prod(meta::CVI, ((z) -> logpdf(m_out, z))::Function, m_in::Any)
            @test isapprox(convert(Distribution, λ), NormalWeightedMeanPrecision(i, 2), atol = 0.5)
        end
    end
end

end
