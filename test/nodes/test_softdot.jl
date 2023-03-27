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
        @test name.(interfaces(node)) === (:out, :θ, :x, :γ)
        @test factorisation(node) === ((1, 2, 3, 4),)
        @test metadata(node) === nothing

        node = make_node(SoftDot, FactorNodeCreationOptions(nothing, 1, nothing))
        @test metadata(node) === 1
    end

    # g(a,b,x) = -a * log(b) + loggamma(a) + (a+1) * (log(scale(x)) - digamma(shape(x))) + b/mean(x)
    #@testset "AverageEnergy" begin
    #    begin
    #        q_out = GammaInverse(2.0, 1.0)
    #        q_α = PointMass(2.0)
    #        q_θ = PointMass(1.0)

    #        marginals = (Marginal(q_out, false, false, nothing), Marginal(q_α, false, false, nothing), Marginal(q_θ, false, false, nothing))

    #        @test score(AverageEnergy(), GammaInverse, Val{(:out, :α, :θ)}(), marginals, nothing) ≈ -0.26835300529540684
    #    end
    #    begin
    #        q_out = GammaInverse(42.0, 42.0)
    #        q_α = PointMass(42.0)
    #        q_θ = PointMass(42.0)

    #        marginals = (Marginal(q_out, false, false, nothing), Marginal(q_α, false, false, nothing), Marginal(q_θ, false, false, nothing))

    #        @test score(AverageEnergy(), GammaInverse, Val{(:out, :α, :θ)}(), marginals, nothing) ≈ -1.433976171558072
    #    end
    #end # testset: AverageEnergy
end # testset
end # module