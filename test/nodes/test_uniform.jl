module UniformNodeTest

using Test
using ReactiveMP
using Random

import ReactiveMP: make_node

@testset "UniformNode" begin
    @testset "Creation" begin
        node = make_node(Uniform)

        @test functionalform(node) === Uniform
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :a, :b)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_a_b,)
        @test metadata(node) === nothing

        node = make_node(Uniform, FactorNodeCreationOptions(nothing, 1, nothing))

        @test metadata(node) === 1
    end

    @testset "Average energy" begin
        node = ReactiveMP.make_node(Uniform)
        for a in 0:0.1:1, b in 1.1:0.1:2.1
            @test isapprox(
                score(AverageEnergy(), Uniform, Val{(:out, :a, :b)},
                    (
                        Marginal(Uniform(a, b), false, false),
                        Marginal(PointMass(a), false, false),
                        Marginal(PointMass(b), false, false)
                    ), nothing), entropy(Uniform(a, b)), rtol = 1e-12)
        end
    end
end
end
