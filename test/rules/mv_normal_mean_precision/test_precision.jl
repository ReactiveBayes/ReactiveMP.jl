module RulesMvNormalMeanPrecisionPrecisionTest

using Test
using ReactiveMP
using Random


import ReactiveMP: @test_rules

@testset "rules:MvNormalMeanPrecision:precision" begin

    @testset "Variational: (q_out::PointMass, q_μ::MultivariateNormalDistributionsFamily)" begin

        @test_rules [ with_float_conversions = false ] MvNormalMeanPrecision(:Λ, Marginalisation) [
            (input = (q_out = PointMass([1.0; 2.0]), q_μ = MvNormalMeanPrecision([3.0; 5.0], [3.0 2.0; 2.0 4.0])), output = Wishart(4.0, cholinv(cholinv([3.0 2.0; 2.0 4.0]) + ([3.0; 5.0] - [1.0; 2.0])*([3.0; 5.0] - [1.0; 2.0])')))
        ]
        @test_rules [ with_float_conversions = false ] MvNormalMeanPrecision(:Λ, Marginalisation) [
            (input = (q_out = PointMass([1.0; 2.0]), q_μ = MvNormalMeanCovariance([3.0; 5.0], cholinv([3.0 2.0; 2.0 4.0]))), output = Wishart(4.0, cholinv(cholinv([3.0 2.0; 2.0 4.0]) + ([3.0; 5.0] - [1.0; 2.0])*([3.0; 5.0] - [1.0; 2.0])')))
        ]
        @test_rules [ with_float_conversions = false ] MvNormalMeanPrecision(:Λ, Marginalisation) [
            (input = (q_out = PointMass([1.0; 2.0]), q_μ = MvNormalWeightedMeanPrecision([19.0; 26.0], [3.0 2.0; 2.0 4.0])), output = Wishart(4.0, cholinv(cholinv([3.0 2.0; 2.0 4.0]) + ([3.0; 5.0] - [1.0; 2.0])*([3.0; 5.0] - [1.0; 2.0])')))
        ]

    end

end



end