module RulesMvNormalMeanPrecisionMeanTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "rules:MvNormalMeanPrecision:mean" begin

    @testset "Belief Propagation: (m_out::PointMass, m_Λ::PointMass)" begin
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = PointMass([-1.0]), m_Λ = PointMass([2.0])), output = MvNormalMeanPrecision([-1.0], [2.0])),
            (input = (m_out = PointMass([1.0]), m_Λ = PointMass([2.0])),  output = MvNormalMeanPrecision([1.0], [2.0])),
            (input = (m_out = PointMass([2.0]), m_Λ = PointMass([1.0])),  output = MvNormalMeanPrecision([2.0], [1.0])),
            (input = (m_out = PointMass([1.0; 3.0]), m_Λ = PointMass([3.0 2.0; 2.0 4.0])),  output = MvNormalMeanPrecision([1.0; 3.0], [3.0 2.0; 2.0 4.0]))
        ]

    end

    @testset "Belief Propagation: (m_out::MultivariateNormalDistributionsFamily, m_Λ::PointMass)" begin

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([2.0; 1.0], [3.0 2.0; 2.0 4.0]),  m_Λ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [0.75 -0.375; -0.375 0.5625]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanCovariance([2.0; 1.0], [0.5 -0.25; -0.25 0.375]),  m_Λ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [0.75 -0.375; -0.375 0.5625]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalWeightedMeanPrecision([8.0; 8.0], [3.0 2.0; 2.0 4.0]),  m_Λ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [0.75 -0.375; -0.375 0.5625])),
        ]

    end

    @testset "Variational: (q_out::Any, q_Λ::Any)" begin

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (q_out = PointMass([2.0; 1.0]), q_Λ = PointMass([2.0 1.0; 5.0 5.0])), output = MvNormalMeanPrecision([2.0; 1.0], [2.0 1.0; 5.0 5.0]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (q_out = MvNormalMeanCovariance([2.0; 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanPrecision([2.0; 1.0], [6.0 4.0; 4.0 8.0]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (q_out = PointMass([2.0; 1.0]), q_Λ = Wishart(5, [3.0 2.0; 2.0 4.0])), output = MvNormalMeanPrecision([2.0; 1.0], [15.0 10.0; 10.0 20.0]))
        ]

    end

    @testset "Structured variational: (m_out::MvNormalMeanPrecision, q_Λ::Any)" begin
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([2.0; 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [0.75 -0.375; -0.375 0.5625]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([2.0; 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = Wishart(5, [3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([2.0; 1.0], [0.6 -0.3; -0.3 0.45]))
        ]
        
    end

end



end