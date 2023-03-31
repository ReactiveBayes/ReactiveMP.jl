module RulesSoftDotOutTest

using Test
using ReactiveMP
#using Random
#using Distributions

import ReactiveMP: @test_rules

# TODO: add combinations of multiple node inputs
@testset "rules:SoftDot:y" begin
    @testset "Variational Message Passing: (q_θ::Any, q_x::Any, q_γ::Any)" begin
        @test_rules [with_float_conversions = true] SoftDot(:y, Marginalisation) [
            (input = (q_θ = PointMass(1.0), q_x = PointMass(2.0), q_γ = PointMass(1.0)), output = NormalMeanPrecision(2.0, 1.0)),
            (input = (q_θ = NormalMeanVariance(1.0, 3.0), q_x = NormalMeanVariance(1.0, 3.0), q_γ = PointMass(1.0)), output = NormalMeanPrecision(1.0, 1.0)),
            (input = (q_θ = MvNormalMeanCovariance([1.0, 1.0], [3.0, 3.0]), q_x = MvNormalMeanCovariance([1.0, 1.0], [3.0, 3.0]), q_γ = PointMass(3.0)), output = NormalMeanPrecision(2.0, 3.0))
        ]
    end
end # testset
end # module
