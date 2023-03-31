module SoftDotNodeTest

using Test
using ReactiveMP
using Random

import ReactiveMP: make_node

@testset "SoftDotNode" begin
    @testset "Creation" begin
        node = make_node(SoftDot)
        @test functionalform(node) === SoftDot
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:y, :θ, :x, :γ)
        @test factorisation(node) === ((1, 2, 3, 4),)
        @test metadata(node) === nothing

        node = make_node(SoftDot, FactorNodeCreationOptions(nothing, 1, nothing))
        @test metadata(node) === 1
    end

    @testset "AverageEnergy" begin
        begin
            q_y = NormalMeanVariance(3.0, 7.0)
            q_θ = NormalMeanVariance(11.0, 13.0)
            q_x = NormalMeanVariance(5.0, 9.0)
            q_γ = GammaShapeRate(3/2, 4242/2)
            marginals = (Marginal(q_y, false, false, nothing), Marginal(q_θ, false, false, nothing), Marginal(q_x, false, false, nothing), Marginal(q_γ, false, false, nothing))

            @test score(AverageEnergy(), SoftDot, Val{(:y, :θ, :x, :γ)}(), marginals, nothing) ≈ 6.3471227390278155
        end

        begin
            q_y = NormalMeanVariance(3.0, 7.0)
            q_θ = MvNormalMeanCovariance([23.0, 29.0], [31.0 37.0; 41.0 43.0])
            q_x = MvNormalMeanCovariance([5.0, 9.0], [11.0 13.0; 17.0 19.0])
            q_γ = GammaShapeRate(3/2, 191032/2)
            marginals = (Marginal(q_y, false, false, nothing), Marginal(q_θ, false, false, nothing), Marginal(q_x, false, false, nothing), Marginal(q_γ, false, false, nothing))

            @test score(AverageEnergy(), SoftDot, Val{(:y, :θ, :x, :γ)}(), marginals, nothing) ≈ 8.15193210352257
        end
    end # testset: AverageEnergy
end # testset
end # module
