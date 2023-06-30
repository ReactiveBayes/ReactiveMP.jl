module ReactiveMPRenderCVITest

using Test
using ReactiveMP
using Random
using StableRNGs
using Distributions
using SpecialFunctions

import ReactiveMP: naturalparams, NaturalParameters, AbstractContinuousGenericLogPdf

struct EmptyOptimizer end

function ReactiveMP.cvi_setup(::EmptyOptimizer, λ)
    return EmptyOptimizer()
end

function ReactiveMP.cvi_update!(::EmptyOptimizer, λ, ∇)
    return vec(λ)
end

function informationmatrix(dist::GammaShapeRate)
    return [polygamma(1, shape(dist)) -1/rate(dist); -1/rate(dist) shape(dist)/rate(dist)^2]
end

@testset "cvi:informationmatrix" begin
    @testset "GammaShapeRate" begin
        seed = 123
        rng = StableRNG(seed)
        meta = CVI(rng, 1, 5000, EmptyOptimizer(), ForwardDiffGrad(), false, false, false)
        for i in 1:100
            for j in 1:100
                gammashaperate = GammaShapeRate(i, j)
                params = vec(naturalparams(gammashaperate))
                estimated_fisher_matrix = ReactiveMP.compute_fisher_matrix(meta, ReactiveMP.GammaNaturalParameters, params)
                @test [1 0; 0 -1]' * estimated_fisher_matrix * [1 0; 0 -1] ≈ informationmatrix(gammashaperate)
            end
        end
    end
end

end
