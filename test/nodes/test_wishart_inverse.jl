module InverseWishartNodeTest

using Test
using ReactiveMP
using Random

import ReactiveMP: make_node

@testset "InverseWishartNode" begin
    @testset "Creation" begin
        node = make_node(InverseWishart)

        @test functionalform(node) === InverseWishart
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :ν, :S)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_ν_S,)
        @test metadata(node) === nothing

        node = make_node(InverseWishart, FactorNodeCreationOptions(nothing, 1, nothing))

        @test metadata(node) === 1
    end

    @testset "AverageEnergy" begin
        begin
            q_out = InverseWishart(2.0, [2.0 0.0; 0.0 2.0])
            q_ν   = PointMass(2.0)
            q_S   = PointMass([2.0 0.0; 0.0 2.0])

            marginals = (
                Marginal(q_out, false, false),
                Marginal(q_ν, false, false),
                Marginal(q_S, false, false)
            )
            @test score(AverageEnergy(), InverseWishart, Val{(:out, :ν, :S)}, marginals, nothing) ≈
                  9.496544113156787
        end

        begin
            S     = [4.3082195553088445 0.4573472347695425 -2.748089173206861; 0.4573472347695425 0.0954087613417567 -0.29586598556052124; -2.748089173206861 -0.29586598556052124 2.9875706318257538]
            ν     = 4.0
            q_out = InverseWishart(ν, S)
            q_ν   = PointMass(ν)
            q_S   = PointMass(S)

            marginals = (
                Marginal(q_out, false, false),
                Marginal(q_ν, false, false),
                Marginal(q_S, false, false)
            )
            @test score(AverageEnergy(), InverseWishart, Val{(:out, :ν, :S)}, marginals, nothing) ≈
                  1.1299587008097587
        end
    end
end
end
