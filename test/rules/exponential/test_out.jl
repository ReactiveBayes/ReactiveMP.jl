module RulesExponentialOutTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:Exponential:out" begin

     @testset "Belief Propagation: (m_in1::PointMass)" begin

        @test_rules [ with_float_conversions = false ] Exponential(:out, Marginalisation) [
            (input = (m_in1 = PointMass(1.0), ), output = PointMass(1.0)),
        ]

    end

end
end