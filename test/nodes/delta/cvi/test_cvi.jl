module ReactiveMPRenderCVITest

using Test
using ReactiveMP
using Random
using StableRNGs

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

@testset "cvi:renderCVI" begin
    @testset "empty optimizer" begin
        m_in = NormalMeanVariance(0, 1)
        m_out = NormalMeanVariance(0, 1)
        η = naturalparams(m_in)
        logp_nc = (z) -> logpdf(m_out, z)
        opt = EmptyOptimizer()
        meta = CVIApproximation(1, 100, opt)
        λ = ReactiveMP.renderCVI(logp_nc, meta.num_iterations, meta.opt, meta.rng, η, m_in)
        @test convert(NormalMeanVariance, λ) ≈ NormalMeanVariance(0, 1)
    end

    @testset "counting optimizer" begin
        m_in = NormalMeanVariance(0, 1)
        m_out = NormalMeanVariance(0, 1)
        η = naturalparams(m_in)
        logp_nc = (z) -> logpdf(m_out, z)
        opt = CountingOptimizer(0)
        meta = CVIApproximation(1, 100, opt)
        λ = ReactiveMP.renderCVI(logp_nc, meta.num_iterations, meta.opt, meta.rng, η, m_in)
        @test convert(NormalMeanVariance, λ) ≈ NormalMeanVariance(0, 1)
        @test opt.num_its === 100
    end

    @testset "counting lambda optimizer" begin
        m_in = NormalMeanVariance(0, 1)
        m_out = NormalMeanVariance(0, 1)
        η = naturalparams(m_in)
        logp_nc = (z) -> logpdf(m_out, z)
        num_its = 0
        opt = (λ, ∇) -> begin
            num_its += 1
            return vec(λ)
        end
        meta = CVIApproximation(1, 100, opt)
        λ = ReactiveMP.renderCVI(logp_nc, meta.num_iterations, meta.opt, meta.rng, η, m_in)
        @test convert(NormalMeanVariance, λ) ≈ NormalMeanVariance(0, 1)
        @test num_its === 100
    end
end

end
