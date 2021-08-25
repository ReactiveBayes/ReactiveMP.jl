module RulesFlowOutTest

using Test
using ReactiveMP
using Random
using Distributions
using LinearAlgebra

import ReactiveMP: @test_rules

@testset "rules:Flow:out" begin

    model = FlowModel( (NiceLayer(PlanarMap(1.0, 2.0, 3.0)), ) )
    meta  = FlowMeta(model)
    Ji1 = inv_jacobian(model, [3.0, -1.5])
    Ji2 = inv_jacobian(model, [-5.0, -1.5])
    J1 = jacobian(model, [3.0, -1.5])
    J2 = jacobian(model, [-5.0, -1.5])

    @testset "Belief Propagation: (m_in::MvNormalMeanCovariance, )" begin

        @test_rules [ with_float_conversions = false, atol = 1e-5 ] Flow(:out, Marginalisation) [
            (input = (m_in = MvNormalMeanCovariance([3.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanCovariance([3.0, 2.5], J1*J1')),
            (input = (m_in = MvNormalMeanCovariance([-5.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanCovariance([-5.0, -7.5], J2*J2')),
            (input = (m_in = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), meta = meta), output = MvNormalMeanCovariance([-5.0, -8.5], J2*diagm([1.0, 2.0])*J2'))
        ]

    end

    @testset "Belief Propagation: (m_in::MvNormalMeanPrecision, )" begin

        @test_rules [ with_float_conversions = false, atol = 1e-5 ] Flow(:out, Marginalisation) [
            (input = (m_in = MvNormalMeanPrecision([3.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanPrecision([3.0, 2.5], Ji1'*Ji1)),
            (input = (m_in = MvNormalMeanPrecision([-5.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanPrecision([-5.0, -7.5], Ji2'*Ji2)),
            (input = (m_in = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 0.5])), meta = meta), output = MvNormalMeanPrecision([-5.0, -8.5], Ji2'*diagm([1.0, 0.5])*Ji2))
        ]

    end

    @testset "Belief Propagation: (m_in::MvNormalWeightedMeanPrecision, )" begin

        @test_rules [ with_float_conversions = false, atol = 1e-5 ] Flow(:out, Marginalisation) [
            (input = (m_in = MvNormalWeightedMeanPrecision([3.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanPrecision([3.0, 2.5], Ji1'*Ji1)),
            (input = (m_in = MvNormalWeightedMeanPrecision([-5.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanPrecision([-5.0, -7.5], Ji2'*Ji2)),
            (input = (m_in = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 0.5])), meta = meta), output = MvNormalMeanPrecision([-5.0, -8.5], Ji2'*diagm([1.0, 0.5])*Ji2))
        ]

    end

end

end