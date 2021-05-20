module RulesNormalMeanVarianceMeanTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @call_rule

@testset "rules:NormalMeanVariance:mean" begin

    @testset "Belief Propagation" begin
        
        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(-1.0), m_v = PointMass(2.0))) == NormalMeanVariance{Float64}(-1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(1.0), m_v = PointMass(2.0)))  == NormalMeanVariance{Float64}(1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(2.0), m_v = PointMass(1.0)))  == NormalMeanVariance{Float64}(2.0, 1.0)

        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(BigFloat(-1.0)), m_v = PointMass(2.0))) == NormalMeanVariance{BigFloat}(-1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(BigFloat(1.0)), m_v = PointMass(2.0)))  == NormalMeanVariance{BigFloat}(1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(BigFloat(2.0)), m_v = PointMass(1.0)))  == NormalMeanVariance{BigFloat}(2.0, 1.0)

        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(-1.0), m_v = PointMass(BigFloat(2.0)))) == NormalMeanVariance{BigFloat}(-1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(1.0), m_v = PointMass(BigFloat(2.0))))  == NormalMeanVariance{BigFloat}(1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(2.0), m_v = PointMass(BigFloat(1.0))))  == NormalMeanVariance{BigFloat}(2.0, 1.0)

        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(BigFloat(-1.0)), m_v = PointMass(BigFloat(2.0)))) == NormalMeanVariance{BigFloat}(-1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(BigFloat(1.0)), m_v = PointMass(BigFloat(2.0))))  == NormalMeanVariance{BigFloat}(1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = PointMass(BigFloat(2.0)), m_v = PointMass(BigFloat(1.0))))  == NormalMeanVariance{BigFloat}(2.0, 1.0)
    end

end



end