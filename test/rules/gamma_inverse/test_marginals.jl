module RulesGammaInverseMarginalsTest

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_marginalrules

# test this set with $ make test testset='rules:gamma_inverse:marginals'
@testitem "marginalrules:GammaInverse" begin
    @testset "out_α_θ: (m_out::GammaInverse, m_α::PointMass, m_θ::PointMass)" begin
        @test_marginalrules [check_type_promotion = false] GammaInverse(:out_α_θ) [
            (input = (m_out = GammaInverse(1.0, 2.0), m_α = PointMass(1.0), m_θ = PointMass(2.0)), output = (out = GammaInverse(3.0, 4.0), α = PointMass(1.0), θ = PointMass(2.0))),
            (input = (m_out = GammaInverse(2.0, 2.0), m_α = PointMass(2.0), m_θ = PointMass(3.0)), output = (out = GammaInverse(5.0, 5.0), α = PointMass(2.0), θ = PointMass(3.0))),
            (input = (m_out = GammaInverse(2.0, 3.0), m_α = PointMass(1.0), m_θ = PointMass(3.0)), output = (out = GammaInverse(4.0, 6.0), α = PointMass(1.0), θ = PointMass(3.0)))
        ]
    end
end
end
