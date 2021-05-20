module RulesNormalMeanVarianceMeanTest

using Test
using ReactiveMP
using Random

import ReactiveMP: @call_rule

@testset "rules:NormalMeanVariance:mean" begin

    @testset "Belief Propagation: (m_μ::PointMass, m_v::PointMass)" begin
        
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

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_v::PointMass)" begin

        m_out_1, m_v_1, result_1 = NormalMeanVariance(0.0, 1.0), PointMass(2.0), NormalMeanVariance(0.0, 3.0)
        m_out_2, m_v_2, result_2 = NormalMeanVariance(-1.0, 1.0), PointMass(1.5), NormalMeanVariance(-1.0, 2.5)
        m_out_3, m_v_3, result_3 = NormalMeanVariance(2.0, 0.1), PointMass(1.2), NormalMeanVariance(2.0, 1.3)

        types = ReactiveMP.union_types(UnivariateNormalDistributionsFamily{Float64})

        for type in types 
            @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = convert(type, m_out_1), m_v = m_v_1)) == result_1
            @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = convert(type, m_out_2), m_v = m_v_2)) == result_2
            @test (@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = convert(type, m_out_3), m_v = m_v_3)) == result_3
        end

        bm_out_1, bm_v_1, bresult_1 = NormalMeanVariance{BigFloat}(0.0, 1.0), PointMass{BigFloat}(2.0), NormalMeanVariance{BigFloat}(0.0, 3.0)
        bm_out_2, bm_v_2, bresult_2 = NormalMeanVariance{BigFloat}(-1.0, 1.0), PointMass{BigFloat}(1.5), NormalMeanVariance{BigFloat}(-1.0, 2.5)
        bm_out_3, bm_v_3, bresult_3 = NormalMeanVariance{BigFloat}(2.0, 0.1), PointMass{BigFloat}(1.2), NormalMeanVariance{BigFloat}(2.0, 1.3)

        btypes = ReactiveMP.union_types(UnivariateNormalDistributionsFamily{BigFloat})

        for type in btypes 
            @test isapprox((@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = convert(type, bm_out_1), m_v = bm_v_1)), bresult_1; atol = 1e-10)
            @test isapprox((@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = convert(type, bm_out_2), m_v = bm_v_2)), bresult_2; atol = 1e-10)
            @test isapprox((@call_rule NormalMeanVariance(:μ, Marginalisation) (m_out = convert(type, bm_out_3), m_v = bm_v_3)), bresult_3; atol = 1e-10)
        end

    end

end



end