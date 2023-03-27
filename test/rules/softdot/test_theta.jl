module RulesSoftDotThetaTest

using Test
using ReactiveMP
#using Random
#using Distributions

import ReactiveMP: @test_rules

# TODO: add combinations of multiple node inputs
@testset "rules:SoftDot:θ" begin
    @testset "Variational Message Passing: (q_y::Any, q_x::Any, q_γ::Any)" begin
        @test_rules [with_float_conversions = true] SoftDot(:θ, Marginalisation) [
            #(input = (q_y = PointMass(1.0), q_x = PointMass(2.0), q_γ = PointMass(1.0)), output = MvNormalMeanPrecision([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
            #(input = (q_y = NormalMeanVariance(1.0, 3.0), q_x = NormalMeanVariance(1.0, 3.0), q_γ = PointMass(1.0)), output = NormalMeanPrecision(1.0, 1.0)),
            #(input = (q_y = MvNormalMeanCovariance([1.0, 1.0], [3.0, 3.0]), q_x = MvNormalMeanCovariance([1.0, 1.0], [3.0, 3.0]), q_γ = PointMass(3.0)), output = NormalMeanPrecision(2.0, 3.0))
        ]
    end
end # testset
end # module
