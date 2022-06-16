module RulesPoissonOutTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:Poisson:out" begin
    @testset "Belief Propagation: (m_l::PointMass)" begin
        @test_rules [with_float_conversions = true] Poisson(:out, Marginalisation) [
            (input = (m_l = PointMass(1.0),), output = Poisson(1.0)),
            (input = (m_l = PointMass(0.2),), output = Poisson(0.2))
        ]
    end

    @testset "Variational Message Passing: (q_l::Gamma)" begin
        @test_rules [with_float_conversions = true, atol = 1e-4] Poisson(:out, Marginalisation) [
            (input = (q_l = Gamma(1.0, 1.0),), output = Poisson(0.56146)),
            (input = (q_l = GammaShapeRate(1.0, 0.5),), output = Poisson(1.12292))
        ]
    end
end
end
