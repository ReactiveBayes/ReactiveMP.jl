module RulesMvNormalMeanPrecisionOutTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "rules:MvNormalMeanPrecision:out" begin

    @testset "Belief Propagation: (m_μ::PointMass, m_Λ::PointMass)" begin
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = PointMass([-1.0]), m_Λ = PointMass([2.0])), output = MvNormalMeanPrecision([-1.0], [2.0])),
            (input = (m_μ = PointMass([1.0]), m_Λ = PointMass([2.0])),  output = MvNormalMeanPrecision([1.0], [2.0])),
            (input = (m_μ = PointMass([2.0]), m_Λ = PointMass([1.0])),  output = MvNormalMeanPrecision([2.0], [1.0])),
            (input = (m_μ = PointMass([1.0; 3.0]), m_Λ = PointMass([3.0 2.0; 2.0 4.0])),  output = MvNormalMeanPrecision([1.0; 3.0], [3.0 2.0; 2.0 4.0]))
        ]

    end

    @testset "Belief Propagation: (m_μ::MultivariateNormalDistributionsFamily, m_Λ::PointMass)" begin

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = MvNormalMeanPrecision([2.0; 1.0], [3.0 2.0; 2.0 4.0]),  m_Λ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [0.75 -0.375; -0.375 0.5625]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = MvNormalMeanCovariance([2.0; 1.0], [0.5 -0.25; -0.25 0.375]),  m_Λ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [0.75 -0.375; -0.375 0.5625]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = MvNormalWeightedMeanPrecision([8.0; 8.0], [3.0 2.0; 2.0 4.0]),  m_Λ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [0.75 -0.375; -0.375 0.5625])),
        ]

    end

    @testset "Variational: (q_μ::Any, q_Λ::Any)" begin

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (q_μ = PointMass([2.0; 1.0]), q_Λ = PointMass([2.0 1.0; 5.0 5.0])), output = MvNormalMeanPrecision([2.0; 1.0], [2.0 1.0; 5.0 5.0]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (q_μ = MvNormalMeanCovariance([2.0; 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanPrecision([2.0; 1.0], [6.0 4.0; 4.0 8.0]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (q_μ = PointMass([2.0; 1.0]), q_Λ = Wishart(5, [3.0 2.0; 2.0 4.0])), output = MvNormalMeanPrecision([2.0; 1.0], [15.0 10.0; 10.0 20.0]))
        ]

    end

    @testset "Structured variational: (m_μ::MvNormalMeanPrecision, q_Λ::Any)" begin
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = MvNormalMeanPrecision([2.0; 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [0.75 -0.375; -0.375 0.5625]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanPrecision(:out, Marginalisation) [
            (input = (m_μ = MvNormalMeanPrecision([2.0; 1.0], [3.0 2.0; 2.0 4.0]), q_Λ = Wishart(5, [3.0 2.0; 2.0 4.0])), output = MvNormalMeanCovariance([2.0; 1.0], [0.6 -0.3; -0.3 0.45]))
        ]
        
    end

end



end