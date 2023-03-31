module RulesSoftDotGammaTest

using Test
using ReactiveMP

import ReactiveMP: @test_rules

# TODO: add combinations of multiple node inputs
@testset "rules:SoftDot:γ" begin
    @testset "Variational Message Passing: (q_y::PointMass, q_θ::PointMass, q_x::PointMass" begin
        @test_rules [with_float_conversions = true] SoftDot(:γ, Marginalisation) [
            (input = (q_y = PointMass(3.0), q_x = PointMass(5.0), q_θ = PointMass(2.0)), output = GammaShapeRate(3/2, 49/2)),
            (input = (q_y = NormalMeanVariance(3.0, 7.0), q_x = NormalMeanVariance(5.0, 9.0), q_θ = PointMass(2.0)), output = GammaShapeRate(3/2, 46.0)),
            (input = (q_y = NormalMeanVariance(3.0, 7.0), q_x = PointMass(2.0), q_θ = NormalMeanVariance(5.0, 9.0)), output = GammaShapeRate(3/2, 46.0)),
        ]
    end

    @testset "Variational Message Passing: (q_y::NormalMeanVariance, q_θ::NormalMeanVariance, q_x::NormalMeanVariance" begin
        @test_rules [with_float_conversions = true] SoftDot(:γ, Marginalisation) [
            (input = (q_y = NormalMeanVariance(3.0, 7.0), q_x = NormalMeanVariance(5.0, 9.0), q_θ = NormalMeanVariance(11.0, 13.0)), output = GammaShapeRate(3/2, 4242/2))
        ]
    end
    @testset "Variational Message Passing: (q_y::NormalMeanVariance, q_θ::MvNormalMeanCovariance, q_x::MvNormalMeanCovariance" begin
        @test_rules [with_float_conversions = true] SoftDot(:γ, Marginalisation) [
            (input = (q_y = NormalMeanVariance(3.0, 7.0), q_x = MvNormalMeanCovariance([5.0, 9.0], [11.0 13.0; 17.0 19.0]), q_θ = MvNormalMeanCovariance([23.0, 29.0], [31.0 37.0; 41.0 43.0])), output = GammaShapeRate(3/2, 191032/2))
        ]
    end

    @testset "Variational Message Passing: (q_θ::INCORRECT, q_x::NormalMeanVariance, q_γ::{<:GammaDistributionsFamily})" begin
        #@test_throws MethodError @call_rule SoftDot(:γ, Marginalisation) (
        #    q_θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]),
        #    q_x = NormalMeanVariance(7.0, 11.0),
        #    q_y = GammaShapeRate(13.0, 5.0)
        #)
    end
end # testset
end # module
