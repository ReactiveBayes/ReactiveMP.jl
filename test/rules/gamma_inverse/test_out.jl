module RulesGammaInverseOutTest

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules

# test this set with $ make test testset='rules:gamma_inverse:out'
@testitem "rules:GammaInverse:out" begin
    @testset "Belief Propagation: (m_α::Any, m_θ::Any)" begin
        @test_rules [check_type_promotion = true] GammaInverse(:out, Marginalisation) [
            (input = (m_α = PointMass(1.0), m_θ = PointMass(2.0)), output = GammaInverse(1.0, 2.0)),
            (input = (m_α = PointMass(3.0), m_θ = PointMass(3.0)), output = GammaInverse(3.0, 3.0)),
            (input = (m_α = PointMass(42.0), m_θ = PointMass(42.0)), output = GammaInverse(42.0, 42.0))
        ]
    end

    @testset "Variational Message Passing: (q_α::Any, q_θ::Any)" begin
        @test_rules [check_type_promotion = true] GammaInverse(:out, Marginalisation) [
            (input = (q_α = PointMass(1.0), q_θ = PointMass(2.0)), output = GammaInverse(1.0, 2.0)),
            (input = (q_α = Gamma(1.0, 1.0), q_θ = Beta(1.0, 1.0)), output = GammaInverse(1.0, 0.5))
        ]
    end
end # testset
end # module
