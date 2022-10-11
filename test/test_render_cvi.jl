module ReactiveMPRenderCVITest

using Test
using ReactiveMP
using Random
using StableRNGs

struct EmptyOptimizer end

function ReactiveMP.cvi_update!(opt::EmptyOptimizer, λ::T, ∇::T) where {T <: NaturalParameters}
    return vec(λ)
end

@testset "cvi:renderCVI" begin
    @testset "empty optimizer" begin
        m_in = NormalMeanVariance(0, 1)
        m_out = NormalMeanVariance(0, 1)
        η = naturalparams(m_in)
        logp_nc = (z) -> logpdf(m_out, z)
        meta = CVIApproximation(100, 100, EmptyOptimizer())
        λ = ReactiveMP.renderCVI(logp_nc, meta.num_iterations, meta.opt, meta.rng, deepcopy(η), m_in)
        @test convert(NormalMeanVariance, λ) ≈ NormalMeanVariance(0, 1)
    end
end
end 