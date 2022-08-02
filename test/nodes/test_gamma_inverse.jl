module InverseWishartNodeTest

using Test
using ReactiveMP
using Random

import ReactiveMP: make_node

@testset "InverseWishartNode" begin
    @testset "Creation" begin
        node = make_node(GammaInverse)

        @test functionalform(node) === GammaInverse
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :α, :θ)
        # XXX ?
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_α_θ,)
        # XXX ?
        @test metadata(node) === nothing

        # XXX ?
        node = make_node(InverseWishart, FactorNodeCreationOptions(nothing, 1, nothing))
        @test metadata(node) === 1
    end

    # TODO: Average Energy testset
    @testset "AverageEnergy" begin
        begin
            q_out = GammaInverse(2.0, 1.0)
            q_α = PointMass(2.0)
            q_θ = PointMass(1.0)

            marginals = (
                Marginal(q_out, false, false),
                Marginal(q_α, false, false),
                Marginal(q_θ, false, false)
            )

            @test score(AverageEnergy(), GammaInverse, Val{(:out, :α, :θ)}, marginals, nothing) ≈ 1.1299587008097587
        end
    end # testset: AverageEnergy
end # testset
end # module
