@testitem "rules:Flow:out" tags = [:rules] begin
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

    @testset "Belief Propagation: (m_in::MvNormalMeanCovariance, ) (Linearization)" begin
        @test_message_update_rule(
            node = Flow, target = :out, algorithm = algorithm, check_type_promotion = false, atol = 1.0e-5,
            cases = [
                (m = (in = MvNormalMeanCovariance([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([3.0, 2.5], J1 * J1'),
                (m = (in = MvNormalMeanCovariance([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([-5.0, -7.5], J2 * J2'),
                (m = (in = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])),),) => MvNormalMeanCovariance([-5.0, -8.5], J2 * diagm([1.0, 2.0]) * J2'),
            ],
        )
    end

    @testset "Belief Propagation: (m_in::MvNormalMeanCovariance, ) (Unscented)" begin
        @test_message_update_rule(
            node = Flow, target = :out, algorithm = algorithmU, check_type_promotion = false, atol = 2.0e-5,
            cases = [
                (m = (in = MvNormalMeanCovariance([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([3.0, 2.5], J1 * J1'),
                (m = (in = MvNormalMeanCovariance([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([-5.0, -7.5], J2 * J2'),
                (m = (in = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])),),) => MvNormalMeanCovariance([-5.0, -8.5], J2 * diagm([1.0, 2.0]) * J2'),
            ],
        )
    end

    @testset "Belief Propagation: (m_in::MvNormalMeanPrecision, ) (Linearization)" begin
        @test_message_update_rule(
            node = Flow, target = :out, algorithm = algorithm, check_type_promotion = false, atol = 1.0e-5,
            cases = [
                (m = (in = MvNormalMeanPrecision([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanPrecision([3.0, 2.5], Ji1' * Ji1),
                (m = (in = MvNormalMeanPrecision([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanPrecision([-5.0, -7.5], Ji2' * Ji2),
                (m = (in = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 0.5])),),) => MvNormalMeanPrecision([-5.0, -8.5], Ji2' * diagm([1.0, 0.5]) * Ji2),
            ],
        )
    end

    @testset "Belief Propagation: (m_in::MvNormalMeanPrecision, ) (Unscented)" begin
        @test_message_update_rule(
            node = Flow, target = :out, algorithm = algorithmU, check_type_promotion = false, atol = 2.0e-5,
            cases = [
                (m = (in = MvNormalMeanPrecision([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([3.0, 2.5], J1 * J1'),
                (m = (in = MvNormalMeanPrecision([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([-5.0, -7.5], J2 * J2'),
                (m = (in = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 0.5])),),) => MvNormalMeanCovariance([-5.0, -8.5], J2 * diagm([1.0, 2.0]) * J2'),
            ],
        )
    end

    @testset "Belief Propagation: (m_in::MvNormalWeightedMeanPrecision, ) (Linearization)" begin
        @test_message_update_rule(
            node = Flow, target = :out, algorithm = algorithm, check_type_promotion = false, atol = 1.0e-5,
            cases = [
                (m = (in = MvNormalWeightedMeanPrecision([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanPrecision([3.0, 2.5], Ji1' * Ji1),
                (m = (in = MvNormalWeightedMeanPrecision([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanPrecision([-5.0, -7.5], Ji2' * Ji2),
                (m = (in = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 0.5])),),) => MvNormalMeanPrecision([-5.0, -8.5], Ji2' * diagm([1.0, 0.5]) * Ji2),
            ],
        )
    end

    @testset "Belief Propagation: (m_in::MvNormalWeightedMeanPrecision, ) (Unscented)" begin
        @test_message_update_rule(
            node = Flow, target = :out, algorithm = algorithmU, check_type_promotion = false, atol = 2.0e-5,
            cases = [
                (m = (in = MvNormalWeightedMeanPrecision([3.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([3.0, 2.5], J1 * J1'),
                (m = (in = MvNormalWeightedMeanPrecision([-5.0, -1.5], diagm(ones(2))),),) => MvNormalMeanCovariance([-5.0, -7.5], J2 * J2'),
                (m = (in = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 0.5])),),) => MvNormalMeanCovariance([-5.0, -8.5], J2 * diagm([1.0, 2.0]) * J2'),
            ],
        )
    end
end

@testitem "rules:Flow:precision and covariance forms agree" tags = [:rules] begin
    using FlowMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, LinearAlgebra

    # A flow that moves every coordinate, so the Jacobians at the input and at the output differ:
    # linearised, the message in precision form is the one in covariance form, inverted.
    model = FlowModel((InputLayer(2), AdditiveCouplingLayer(PlanarFlow(); permute = false), PermutationLayer(PermutationMatrix([2, 1])), AdditiveCouplingLayer(PlanarFlow(); permute = false)))
    algorithm = FlowApproximation(compile(model, collect(range(-0.4, 0.6; length = nr_params(model)))))
    μ, Σ = [0.5, -0.3], [0.6 0.1; 0.1 0.4]
    for (target, other) in ((:out, :in), (:in, :out))
        covariance_form = getresult(call_message_update_rule(Flow, target; m = NamedTuple{(other,)}((MvNormalMeanCovariance(μ, Σ),)), algorithm))
        for precision_form in (MvNormalMeanPrecision(μ, inv(Σ)), MvNormalWeightedMeanPrecision(inv(Σ) * μ, inv(Σ)))
            message = getresult(call_message_update_rule(Flow, target; m = NamedTuple{(other,)}((precision_form,)), algorithm))
            @test mean(message) ≈ mean(covariance_form)
            @test cov(message) ≈ cov(covariance_form)
        end
    end
end
