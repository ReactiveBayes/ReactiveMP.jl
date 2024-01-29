module RulesFlowOutTest

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, LinearAlgebra

import ReactiveMP: @test_rules
using ReactiveMP: jacobian, inv_jacobian

@testitem "rules:Flow:out" begin
    params = [1.0, 2.0, 3.0]
    model = FlowModel(2, (AdditiveCouplingLayer(PlanarFlow(); permute = false),))
    compiled_model = compile(model, params)
    meta = FlowMeta(compiled_model)
    metaU = FlowMeta(compiled_model, Unscented(2))
    Ji1 = inv_jacobian(compiled_model, [3.0, -1.5])
    Ji2 = inv_jacobian(compiled_model, [-5.0, -1.5])
    J1 = jacobian(compiled_model, [3.0, -1.5])
    J2 = jacobian(compiled_model, [-5.0, -1.5])

    @testset "Belief Propagation: (m_in::MvNormalMeanCovariance, ) (Linearization)" begin
        @test_rules [check_type_promotion = false, atol = 1e-5] Flow(:out, Marginalisation) [
            (input = (m_in = MvNormalMeanCovariance([3.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanCovariance([3.0, 2.5], J1 * J1')),
            (input = (m_in = MvNormalMeanCovariance([-5.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanCovariance([-5.0, -7.5], J2 * J2')),
            (input = (m_in = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), meta = meta), output = MvNormalMeanCovariance([-5.0, -8.5], J2 * diagm([1.0, 2.0]) * J2'))
        ]
    end

    @testset "Belief Propagation: (m_in::MvNormalMeanCovariance, ) (Unscented)" begin
        @test_rules [check_type_promotion = false, atol = 2e-5] Flow(:out, Marginalisation) [
            (input = (m_in = MvNormalMeanCovariance([3.0, -1.5], diagm(ones(2))), meta = metaU), output = MvNormalMeanCovariance([3.0, 2.5], J1 * J1')),
            (input = (m_in = MvNormalMeanCovariance([-5.0, -1.5], diagm(ones(2))), meta = metaU), output = MvNormalMeanCovariance([-5.0, -7.5], J2 * J2')),
            (input = (m_in = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), meta = metaU), output = MvNormalMeanCovariance([-5.0, -8.5], J2 * diagm([1.0, 2.0]) * J2'))
        ]
    end

    @testset "Belief Propagation: (m_in::MvNormalMeanPrecision, ) (Linearization)" begin
        @test_rules [check_type_promotion = false, atol = 1e-5] Flow(:out, Marginalisation) [
            (input = (m_in = MvNormalMeanPrecision([3.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanPrecision([3.0, 2.5], Ji1' * Ji1)),
            (input = (m_in = MvNormalMeanPrecision([-5.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanPrecision([-5.0, -7.5], Ji2' * Ji2)),
            (input = (m_in = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 0.5])), meta = meta), output = MvNormalMeanPrecision([-5.0, -8.5], Ji2' * diagm([1.0, 0.5]) * Ji2))
        ]
    end

    @testset "Belief Propagation: (m_in::MvNormalMeanPrecision, ) (Unscented)" begin
        @test_rules [check_type_promotion = false, atol = 2e-5] Flow(:out, Marginalisation) [
            (input = (m_in = MvNormalMeanPrecision([3.0, -1.5], diagm(ones(2))), meta = metaU), output = MvNormalMeanCovariance([3.0, 2.5], J1 * J1')),
            (input = (m_in = MvNormalMeanPrecision([-5.0, -1.5], diagm(ones(2))), meta = metaU), output = MvNormalMeanCovariance([-5.0, -7.5], J2 * J2')),
            (input = (m_in = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 0.5])), meta = metaU), output = MvNormalMeanCovariance([-5.0, -8.5], J2 * diagm([1.0, 2.0]) * J2'))
        ]
    end

    @testset "Belief Propagation: (m_in::MvNormalWeightedMeanPrecision, ) (Linearization)" begin
        @test_rules [check_type_promotion = false, atol = 1e-5] Flow(:out, Marginalisation) [
            (input = (m_in = MvNormalWeightedMeanPrecision([3.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanPrecision([3.0, 2.5], Ji1' * Ji1)),
            (input = (m_in = MvNormalWeightedMeanPrecision([-5.0, -1.5], diagm(ones(2))), meta = meta), output = MvNormalMeanPrecision([-5.0, -7.5], Ji2' * Ji2)),
            (
                input = (m_in = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 0.5])), meta = meta),
                output = MvNormalMeanPrecision([-5.0, -8.5], Ji2' * diagm([1.0, 0.5]) * Ji2)
            )
        ]
    end

    @testset "Belief Propagation: (m_in::MvNormalWeightedMeanPrecision, ) (Unscented)" begin
        @test_rules [check_type_promotion = false, atol = 2e-5] Flow(:out, Marginalisation) [
            (input = (m_in = MvNormalWeightedMeanPrecision([3.0, -1.5], diagm(ones(2))), meta = metaU), output = MvNormalMeanCovariance([3.0, 2.5], J1 * J1')),
            (input = (m_in = MvNormalWeightedMeanPrecision([-5.0, -1.5], diagm(ones(2))), meta = metaU), output = MvNormalMeanCovariance([-5.0, -7.5], J2 * J2')),
            (
                input = (m_in = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 0.5])), meta = metaU),
                output = MvNormalMeanCovariance([-5.0, -8.5], J2 * diagm([1.0, 2.0]) * J2')
            )
        ]
    end
end

end
