module ProbitNodeTest

using Test
using ReactiveMP
using Random

@testset "ProbitNode" begin
    @testset "Creation" begin
        node = make_node(Probit)

        @test functionalform(node) === Probit
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :in)
        @test factorisation(node) === ((1, 2),)
        @test metadata(node) === ProbitMeta(32)

        node = make_node(Probit, FactorNodeCreationOptions(nothing, 1, nothing))

        @test functionalform(node) === Probit
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :in)
        @test factorisation(node) === ((1, 2),)
        @test metadata(node) === 1
    end

    @testset "Average energy" begin
        node = make_node(Probit)

        @test score(AverageEnergy(),
            Probit,
            Val{(:out, :in)},
            (Marginal(Bernoulli(1), false, false), Marginal(NormalMeanVariance(0.0, 1.0), false, false)),
            ProbitMeta()) ≈ 1.0

        @test score(AverageEnergy(),
            Probit,
            Val{(:out, :in)},
            (Marginal(PointMass(1), false, false), Marginal(NormalMeanVariance(0.0, 1.0), false, false)),
            ProbitMeta(100)) ≈ 1.0

        for k in 0:0.1:1
            @test score(AverageEnergy(),
                Probit,
                Val{(:out, :in)},
                (Marginal(Bernoulli(k), false, false), Marginal(NormalMeanVariance(0.0, 1.0), false, false)),
                ProbitMeta(100)) ≈ 1.0
        end
    end
end

end
