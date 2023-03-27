module RulesSoftDotXTest

using Test
using ReactiveMP

import ReactiveMP: @test_rules

# TODO: add combinations of multiple node inputs
@testset "rules:SoftDot:x" begin
    @testset "Variational Message Passing: (q_y::Any, q_θ::Any, q_γ::Any)" begin
        @test_rules [with_float_conversions = true] SoftDot(:x, Marginalisation) [
            (input = (q_y = PointMass(3.0), q_θ = PointMass(5.0), q_γ = PointMass(2.0)), output = NormalWeightedMeanPrecision(30.0, 50.0)),
            (input = (q_y = NormalMeanVariance(3.0, 7.0), q_θ = NormalMeanVariance(5.0, 9.0), q_γ = PointMass(2.0)), output = NormalWeightedMeanPrecision(30.0, 68.0)),
            (input = (q_y = NormalMeanVariance(3.0, 7.0), q_θ = MvNormalMeanCovariance([5.0, 5.0], [11.0 13.0; 17.0 19.0]), q_γ = PointMass(2.0)), output = MvNormalWeightedMeanPrecision([30.0, 30.0], [72.0 76.0; 84.0 88.0]))
        ]
    end
end # testset
end # module
