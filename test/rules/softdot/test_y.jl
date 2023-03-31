module RulesSoftDotOutTest

using Test
using ReactiveMP
#using Random
#using Distributions

import ReactiveMP: @test_rules

# TODO: add combinations of multiple node inputs
@testset "rules:SoftDot:y" begin
    @testset "Variational Message Passing: (q_θ::PointMass, q_x::PointMass, q_γ::{<:GammaDistributionsFamily})" begin
        @test_rules [with_float_conversions = true] SoftDot(:y, Marginalisation) [
            (input = (q_θ = PointMass(3.0), q_x = PointMass(11.0), q_γ = GammaShapeRate(7.0, 5.0)), output = NormalMeanPrecision(33.0, 1.4)),
            (input = (q_θ = PointMass(3.0), q_x = PointMass(11.0), q_γ = GammaShapeScale(7.0, 5.0)), output = NormalMeanPrecision(33.0, 35.0))
        ]
    end

    @testset "Variational Message Passing: (q_θ::NormalMeanVariance, q_x::NormalMeanVariance, q_γ::{<:GammaDistributionsFamily})" begin
        @test_rules [with_float_conversions = true] SoftDot(:y, Marginalisation) [
            (input = (q_θ = NormalMeanVariance(3.0, 17.0), q_x = NormalMeanVariance(7.0, 11.0), q_γ = GammaShapeRate(13.0, 5.0)), output = NormalMeanPrecision(21.0, 2.6)),
            (input = (q_θ = NormalMeanVariance(3.0, 17.0), q_x = NormalMeanVariance(7.0, 11.0), q_γ = GammaShapeScale(13.0, 5.0)), output = NormalMeanPrecision(21.0, 65.0))
        ]
    end

    @testset "Variational Message Passing: (q_θ::MvNormalMeanCovariance, q_x::MvNormalMeanCovariance, q_γ::{<:GammaDistributionsFamily})" begin
        @test_rules [with_float_conversions = true] SoftDot(:y, Marginalisation) [
            (input = (q_θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_x = MvNormalMeanCovariance([17.0, 19.0], [23.0, 29.0]), q_γ = GammaShapeRate(31.0, 5.0)), output = NormalMeanPrecision(184.0, 6.2)),
            (input = (q_θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]), q_x = MvNormalMeanCovariance([17.0, 19.0], [23.0, 29.0]), q_γ = GammaShapeScale(31.0, 5.0)), output = NormalMeanPrecision(184.0, 155.0))
        ]
    end

    @testset "Variational Message Passing: (q_θ::INCORRECT, q_x::NormalMeanVariance, q_γ::{<:GammaDistributionsFamily})" begin
        @test_throws MethodError @call_rule SoftDot(:y, Marginalisation) (
            q_θ = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]),
            q_x = NormalMeanVariance(7.0, 11.0),
            q_γ = GammaShapeRate(13.0, 5.0)
        )
    end

    @testset "Variational Message Passing: (q_θ::NormalMeanVariance, q_x::INCORRECT, q_γ::{<:GammaDistributionsFamily})" begin
        @test_throws MethodError @call_rule SoftDot(:y, Marginalisation) (
            q_θ = NormalMeanVariance(7.0, 11.0),
            q_x = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]),
            q_γ = GammaShapeRate(13.0, 5.0)
        )
    end

    @testset "Variational Message Passing: (q_θ::INCORRECT, q_x::MvNormalMeanCovariance, q_γ::{<:GammaDistributionsFamily})" begin
        @test_throws MethodError @call_rule SoftDot(:y, Marginalisation) (
            q_θ = PointMass(7.0),
            q_x = MvNormalMeanCovariance([3.0, 7.0], [11.0, 13.0]),
            q_γ = GammaShapeRate(13.0, 5.0)
        )
    end
end # testset
end # module
