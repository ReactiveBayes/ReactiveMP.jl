module PoissonNodeTest

using Test
using ReactiveMP
using Random

import ReactiveMP: make_node

@testset "PoissonNode" begin
    @testset "Creation" begin
        node = make_node(Poisson)

        @test functionalform(node) === Poisson
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :l)
        @test factorisation(node) === ((1, 2),)
        @test localmarginalnames(node) === (:out_l,)
        @test metadata(node) === nothing

        node = make_node(Poisson, FactorNodeCreationOptions(nothing, 1, nothing))
        
        @test metadata(node) === 1
    end

    @testset "Average energy" begin
        node = make_node(Poisson)
        
        for k in 1:100
            println(score(AverageEnergy(),
            Poisson,
            Val{(:out, :l)},
            (Marginal(Poisson(k), false, false), Marginal(PointMass(k), false, false)), nothing))
            println(entropy(Poisson(k)))
            @test score(AverageEnergy(),
            Poisson,
            Val{(:out, :l)},
            (Marginal(Poisson(k), false, false), Marginal(PointMass(k), false, false)), nothing) ≈ entropy(Poisson(k))
        end
    end
end
end