# From v6's `test/rules/flow/in_tests.jl`.

@testitem "rules:Flow:in" tags = [:rules] begin
    using FlowMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, MessagePassingRulesApproximations, BayesBase, ExponentialFamily, Distributions, LinearAlgebra
    using FlowMessagePassingRules: jacobian, inv_jacobian

    params = [1.0, 2.0, 3.0]
    model = FlowModel(2, (AdditiveCouplingLayer(PlanarFlow(); permute = false),))
    compiled_model = compile(model, params)
    algorithm = FlowApproximation(compiled_model)
    algorithmU = FlowApproximation(compiled_model; method = Unscented(2))
    Ji1 = inv_jacobian(compiled_model, [3.0, -1.5])
    Ji2 = inv_jacobian(compiled_model, [-5.0, -1.5])
    J1 = jacobian(compiled_model, [3.0, -1.5])
    J2 = jacobian(compiled_model, [-5.0, -1.5])

    @testset "Belief Propagation: (m_out::MvNormalMeanCovariance, ) (Linearization)" begin
        @test_message_update_rule(
            node = Flow, target = :in, algorithm = algorithm, check_type_promotion = false, atol = 1.0e-5,
            cases = [
                (m = (out = MvNormalMeanCovariance([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([3.0, -5.5], Ji1 * Ji1'),
                (m = (out = MvNormalMeanCovariance([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([-5.0, 4.5], Ji2 * Ji2'),
                (m = (out = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])),),) => MvNormalMeanCovariance([-5.0, 3.5], Ji2 * diagm([1.0, 2.0]) * Ji2'),
            ],
        )
    end

    @testset "Belief Propagation: (m_out::MvNormalMeanCovariance, ) (Unscented)" begin
        @test_message_update_rule(
            node = Flow, target = :in, algorithm = algorithmU, check_type_promotion = false, atol = 2.0e-5,
            cases = [
                (m = (out = MvNormalMeanCovariance([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([3.0, -5.5], Ji1 * Ji1'),
                (m = (out = MvNormalMeanCovariance([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([-5.0, 4.5], Ji2 * Ji2'),
                (m = (out = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])),),) => MvNormalMeanCovariance([-5.0, 3.5], Ji2 * diagm([1.0, 2.0]) * Ji2'),
            ],
        )
    end

    @testset "Belief Propagation: (m_out::MvNormalMeanPrecision, ) (Linearization)" begin
        @test_message_update_rule(
            node = Flow, target = :in, algorithm = algorithm, check_type_promotion = false, atol = 1.0e-5,
            cases = [
                (m = (out = MvNormalMeanPrecision([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanPrecision([3.0, -5.5], J1' * J1),
                (m = (out = MvNormalMeanPrecision([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanPrecision([-5.0, 4.5], J2' * J2),
                (m = (out = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 0.5])),),) => MvNormalMeanPrecision([-5.0, 3.5], J2' * diagm([1.0, 0.5]) * J2),
            ],
        )
    end

    @testset "Belief Propagation: (m_out::MvNormalMeanPrecision, ) (Unscented)" begin
        @test_message_update_rule(
            node = Flow, target = :in, algorithm = algorithmU, check_type_promotion = false, atol = 2.0e-5,
            cases = [
                (m = (out = MvNormalMeanPrecision([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([3.0, -5.5], Ji1 * Ji1'),
                (m = (out = MvNormalMeanPrecision([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([-5.0, 4.5], Ji2 * Ji2'),
                (m = (out = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 0.5])),),) => MvNormalMeanCovariance([-5.0, 3.5], Ji2 * diagm([1.0, 2.0]) * Ji2'),
            ],
        )
    end

    @testset "Belief Propagation: (m_out::MvNormalWeightedMeanPrecision, ) (Linearization)" begin
        @test_message_update_rule(
            node = Flow, target = :in, algorithm = algorithm, check_type_promotion = false, atol = 1.0e-5,
            cases = [
                (m = (out = MvNormalWeightedMeanPrecision([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanPrecision([3.0, -5.5], J1' * J1),
                (m = (out = MvNormalWeightedMeanPrecision([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanPrecision([-5.0, 4.5], J2' * J2),
                (m = (out = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 0.5])),),) => MvNormalMeanPrecision([-5.0, 3.5], J2' * diagm([1.0, 0.5]) * J2),
            ],
        )
    end

    @testset "Belief Propagation: (m_out::MvNormalWeightedMeanPrecision, ) (Unscented)" begin
        @test_message_update_rule(
            node = Flow, target = :in, algorithm = algorithmU, check_type_promotion = false, atol = 2.0e-5,
            cases = [
                (m = (out = MvNormalWeightedMeanPrecision([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([3.0, -5.5], Ji1 * Ji1'),
                (m = (out = MvNormalWeightedMeanPrecision([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([-5.0, 4.5], Ji2 * Ji2'),
                (m = (out = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 0.5])),),) => MvNormalMeanCovariance([-5.0, 3.5], Ji2 * diagm([1.0, 2.0]) * Ji2'),
            ],
        )
    end
end
