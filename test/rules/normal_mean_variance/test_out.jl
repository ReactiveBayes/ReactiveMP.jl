module RulesNormalMeanVarianceOutTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @call_rule

@testset "rules:NormalMeanVariance:out" begin

    @testset "Belief Propagation: (m_μ::PointMass, m_v::PointMass)" begin
        
        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(-1.0), m_v = PointMass(2.0))) == NormalMeanVariance{Float64}(-1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(1.0), m_v = PointMass(2.0)))  == NormalMeanVariance{Float64}(1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(2.0), m_v = PointMass(1.0)))  == NormalMeanVariance{Float64}(2.0, 1.0)

        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(BigFloat(-1.0)), m_v = PointMass(2.0))) == NormalMeanVariance{BigFloat}(-1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(BigFloat(1.0)), m_v = PointMass(2.0)))  == NormalMeanVariance{BigFloat}(1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(BigFloat(2.0)), m_v = PointMass(1.0)))  == NormalMeanVariance{BigFloat}(2.0, 1.0)

        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(-1.0), m_v = PointMass(BigFloat(2.0)))) == NormalMeanVariance{BigFloat}(-1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(1.0), m_v = PointMass(BigFloat(2.0))))  == NormalMeanVariance{BigFloat}(1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(2.0), m_v = PointMass(BigFloat(1.0))))  == NormalMeanVariance{BigFloat}(2.0, 1.0)

        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(BigFloat(-1.0)), m_v = PointMass(BigFloat(2.0)))) == NormalMeanVariance{BigFloat}(-1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(BigFloat(1.0)), m_v = PointMass(BigFloat(2.0))))  == NormalMeanVariance{BigFloat}(1.0, 2.0)
        @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = PointMass(BigFloat(2.0)), m_v = PointMass(BigFloat(1.0))))  == NormalMeanVariance{BigFloat}(2.0, 1.0)

    end

    @testset "Belief Propagation: (m_μ::UnivariateNormalDistributionsFamily, m_v::PointMass)" begin

        m_μ_1    = NormalMeanVariance(0.0, 1.0)
        m_v_1    = PointMass(2.0)
        result_1 = NormalMeanVariance(0.0, 3.0)

        types = ReactiveMP.union_types(UnivariateNormalDistributionsFamily)

        for type in types 
            @test (@call_rule NormalMeanVariance(:out, Marginalisation) (m_μ = convert(type, m_μ_1), m_v = m_v_1)) == result_1
        end

    end

end



end