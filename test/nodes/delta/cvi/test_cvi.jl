module ReactiveMPRenderCVITest

using Test
using ReactiveMP
using Random
using StableRNGs
using Distributions

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

@testset "cvi:render_cvi" begin
    @testset "empty optimizer" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            opt = EmptyOptimizer()
            meta = CVIApproximation(1, 100, opt)
            λ = ReactiveMP.render_cvi(meta, (z) -> logpdf(m_out, z), m_in)
            @test all(mean_var(convert(Distribution, λ)) .== mean_var(m_in))
        end
    end

    @testset "counting optimizer" begin
        for (m_in, m_out) in ((NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)), (GammaShapeRate(2, 2), GammaShapeRate(2, 2)), (Bernoulli(0.5), Bernoulli(0.5)))
            opt = CountingOptimizer(0)
            meta = CVIApproximation(1, 100, opt)
            λ = ReactiveMP.render_cvi(meta, (z) -> logpdf(m_out, z), m_in)
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
            meta = CVIApproximation(1, 100, opt)
            λ = ReactiveMP.render_cvi(meta, (z) -> logpdf(m_out, z), m_in)
            @test all(mean_var(convert(Distribution, λ)) .== mean_var(m_in))
            @test num_its === 100
        end
    end
end

end
