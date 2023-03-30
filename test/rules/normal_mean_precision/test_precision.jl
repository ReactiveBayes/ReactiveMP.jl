module RulesNormalMeanPrecisionPrecisionTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "rules:NormalMeanPrecision:precision" begin
    @testset "Variational: (q_out::Any, q_μ::Any)" begin
        @test_rules [check_type_promotion = true] NormalMeanPrecision(:τ, Marginalisation) [
            (input = (q_out = PointMass(-1.0), q_μ = PointMass(2.0)), output = Gamma(1.5, 2.0 / 9.0)),
            (input = (q_out = PointMass(1.0), q_μ = PointMass(2.0)), output = Gamma(1.5, 2.0)),
            (input = (q_out = PointMass(2.0), q_μ = PointMass(1.0)), output = Gamma(1.5, 2.0))
        ]

        @test_rules [check_type_promotion = true] NormalMeanPrecision(:τ, Marginalisation) [
            (input = (q_out = NormalMeanVariance(-1.0, 2.0), q_μ = PointMass(2.0)), output = Gamma(1.5, 2.0 / 11.0)),
            (input = (q_out = NormalMeanPrecision(1.0, 4.0), q_μ = PointMass(3.0)), output = Gamma(1.5, 2.0 / 4.25)),
            (input = (q_out = NormalWeightedMeanPrecision(2.0, 4.0), q_μ = PointMass(1.0)), output = Gamma(1.5, 4.0))
        ]

        @test_rules [check_type_promotion = true] NormalMeanPrecision(:τ, Marginalisation) [
            (input = (q_out = PointMass(2.0), q_μ = NormalMeanVariance(-1.0, 2.0)), output = Gamma(1.5, 2.0 / 11.0)),
            (input = (q_out = PointMass(3.0), q_μ = NormalMeanPrecision(1.0, 4.0)), output = Gamma(1.5, 2.0 / 4.25)),
            (input = (q_out = PointMass(1.0), q_μ = NormalWeightedMeanPrecision(2.0, 4.0)), output = Gamma(1.5, 4.0))
        ]

        @test_rules [check_type_promotion = true] NormalMeanPrecision(:τ, Marginalisation) [
            (input = (q_out = NormalMeanVariance(2.0, 4.0), q_μ = NormalMeanVariance(-1.0, 2.0)), output = Gamma(1.5, 2.0 / 15.0)),
            (input = (q_out = NormalMeanPrecision(3.0, 3.0), q_μ = NormalMeanPrecision(1.0, 4.0)), output = Gamma(1.5, 24.0 / 55.0)),
            (input = (q_out = NormalWeightedMeanPrecision(1.0, 1.0), q_μ = NormalWeightedMeanPrecision(2.0, 4.0)), output = Gamma(1.5, 2.0 / 1.5))
        ]
    end

    @testset "Variational: (q_out_μ::Any)" begin
        @test_rules [check_type_promotion = true] NormalMeanPrecision(:τ, Marginalisation) [
            (input = (q_out_μ = MvNormalMeanCovariance([1.0, 1.0], [1.0 0.0; 0.0 1.0]),), output = Gamma(1.5, 1.0)),
            (input = (q_out_μ = MvNormalMeanCovariance([2.0, 3.0], [2.0 -0.1; -0.1 3.0]),), output = Gamma(1.5, 2.0 / 6.2)),
            (input = (q_out_μ = MvNormalMeanCovariance([4.0, 1.0], [4.0 1.0; 1.0 9.0]),), output = Gamma(1.5, 2.0 / 20.0))
        ]

        @test_rules [check_type_promotion = true] NormalMeanPrecision(:τ, Marginalisation) [
            (input = (q_out_μ = MvNormalMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0]),), output = Gamma(1.5, 1.0)),
            (input = (q_out_μ = MvNormalMeanPrecision([2.0, 3.0], [2.0 -0.1; -0.1 3.0]),), output = Gamma(1.5, 1198.0 / 1079.0)),
            (input = (q_out_μ = MvNormalMeanPrecision([4.0, 1.0], [4.0 1.0; 1.0 9.0]),), output = Gamma(1.5, 70.0 / 330.0))
        ]

        @test_rules [check_type_promotion = true] NormalMeanPrecision(:τ, Marginalisation) [
            (input = (q_out_μ = MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0]),), output = Gamma(1.5, 1.0)),
            (input = (q_out_μ = MvNormalWeightedMeanPrecision([4.0, 1.0], [4.0 1.0; 1.0 9.0]),), output = Gamma(1.5, 1.4))
        ]
    end
end

end
