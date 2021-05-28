module RulesMvNormalMeanCovarianceMeanTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @test_rules

@testset "rules:MvNormalMeanCovariance:mean" begin

    @testset "Belief Propagation: (m_out::PointMass, m_Σ::PointMass)" begin
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (m_out = PointMass([-1.0]), m_Σ = PointMass([2.0])), output = MvNormalMeanCovariance([-1.0], [2.0])),
            (input = (m_out = PointMass([1.0]), m_Σ = PointMass([2.0])),  output = MvNormalMeanCovariance([1.0], [2.0])),
            (input = (m_out = PointMass([2.0]), m_Σ = PointMass([1.0])),  output = MvNormalMeanCovariance([2.0], [1.0])),
            (input = (m_out = PointMass([1.0; 3.0]), m_Σ = PointMass([3.0 2.0; 2.0 4.0])),  output = MvNormalMeanCovariance([1.0; 3.0], [3.0 2.0; 2.0 4.0]))
        ]

    end

    @testset "Belief Propagation: (m_out::MultivariateNormalDistributionsFamily, m_Σ::PointMass)" begin

        @test_rules [ with_float_conversions = true ] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanCovariance([2.0; 1.0], [3.0 2.0; 2.0 4.0]),  m_Σ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [9.0 6.0; 6.0 12.0]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([2.0; 1.0], [0.5 -0.25; -0.25 0.375]),  m_Σ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [9.0 6.0; 6.0 12.0]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (m_out = MvNormalWeightedMeanPrecision([0.75; -0.125], [0.5 -0.25; -0.25 0.375]),  m_Σ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [9.0 6.0; 6.0 12.0])),
        ]

    end

    @testset "Variational: (q_out::Any, q_Σ::Any)" begin

        @test_rules [ with_float_conversions = true ] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (q_out = PointMass([2.0; 1.0]), q_Σ = PointMass([2.0 1.0; 5.0 5.0])), output = MvNormalMeanCovariance([2.0; 1.0], [2.0 1.0; 5.0 5.0]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (q_out = MvNormalMeanCovariance([2.0; 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [6.0 4.0; 4.0 8.0]))
        ]

    end

    @testset "Structured variational: (m_out::PointMass, q_Σ::Any)" begin
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (m_out = PointMass([2.0; 1.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [6.0 4.0; 4.0 8.0]))
        ]
        
    end

    @testset "Structured variational: (m_out::MultivariateNormalDistributionsFamily, q_Σ::Any)" begin
        
        @test_rules [ with_float_conversions = true ] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanCovariance([2.0; 1.0], [3.0 2.0; 2.0 4.0]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [9.0 6.0; 6.0 12.0]))
        ]
       
        @test_rules [ with_float_conversions = true ] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([2.0; 1.0], [0.5 -0.25; -0.25 0.375]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [9.0 6.0; 6.0 12.0]))
        ]

        @test_rules [ with_float_conversions = true ] MvNormalMeanCovariance(:μ, Marginalisation) [
            (input = (m_out = MvNormalWeightedMeanPrecision([0.75; -0.125], [0.5 -0.25; -0.25 0.375]), q_Σ = PointMass([6.0 4.0; 4.0 8.0])), output = MvNormalMeanCovariance([2.0; 1.0], [9.0 6.0; 6.0 12.0]))
        ]
        
    end

end

end