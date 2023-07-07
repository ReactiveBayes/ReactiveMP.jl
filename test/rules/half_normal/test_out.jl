module RulesHalfNormalOutTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

@testset "rules:HalfNormal:out" begin
    @testset "Variational Message Passing: (q_v::Any)" begin
        @test_rules [check_type_promotion = false] HalfNormal(:out, Marginalisation) [
            (input = (q_v = PointMass(1.0),), output = Truncated(Normal(0.0, 1.0), 0.0, Inf)), (input = (q_v = PointMass(100),), output = Truncated(Normal(0.0, 10.0), 0.0, Inf))
        ]
    end
end # testset
end # module
