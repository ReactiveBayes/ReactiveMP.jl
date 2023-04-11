module RulesPoissonMarginalsTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_marginalrules

@testset "marginalrules:Poisson" begin
    @testset "out_l: (m_out::PointMass, m_l::Gamma)" begin
        @test_marginalrules [check_type_promotion = true] Poisson(:out_l) [
            (input = (m_out = PointMass(1.0), m_l = Gamma(2.0, 1.0)), output = (out = PointMass(1.0), l = Gamma(3.0, 0.5))),
            (input = (m_out = PointMass(1.0), m_l = Gamma(4.0, 2.0)), output = (out = PointMass(1.0), l = Gamma(5.0, 2//3))),
            (input = (m_out = PointMass(0.0), m_l = Gamma(1.0, 2.0)), output = (out = PointMass(0.0), l = Gamma(1.0, 2//3)))
        ]
    end
end
end
