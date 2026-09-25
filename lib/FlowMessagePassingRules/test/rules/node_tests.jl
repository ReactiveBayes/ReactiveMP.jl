# The node's algorithm, `FlowApproximation(model; method)`, and the missing rule under
# `DefaultAlgorithm()`.

@testitem "rules:Flow:node" tags = [:rules] begin
    using FlowMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, MessagePassingRulesApproximations, BayesBase, ExponentialFamily, Distributions, LinearAlgebra
    using FlowMessagePassingRules: getmodel, getmethod, AbstractCompiledFlowModel
    using MessagePassingRulesApproximations: getL, getα, getβ, getκ, getλ

    @testset "FlowApproximation: default method" begin
        algorithm = FlowApproximation(compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow()),))))
        @test algorithm isa MessagePassingRulesBase.AbstractAlgorithm
        @test typeof(getmodel(algorithm)) <: AbstractCompiledFlowModel
        @test getmodel(algorithm) === algorithm.model
        @test getmethod(algorithm) === algorithm.method
        @test getmethod(algorithm) isa Linearization
    end

    @testset "FlowApproximation: Linearization" begin
        algorithm = FlowApproximation(compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow()),))); method = Linearization())
        @test typeof(getmodel(algorithm)) <: AbstractCompiledFlowModel
        @test getmodel(algorithm) === algorithm.model
        @test getmethod(algorithm) === algorithm.method
        @test getmethod(algorithm) isa Linearization
        # The positional constructor is the same.
        @test FlowApproximation(getmodel(algorithm), Linearization()) isa typeof(algorithm)
    end

    @testset "FlowApproximation: Unscented" begin
        algorithm = FlowApproximation(compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow()),))); method = Unscented(3))
        @test typeof(getmodel(algorithm)) <: AbstractCompiledFlowModel
        @test getmodel(algorithm) === algorithm.model
        @test getmethod(algorithm) === algorithm.method
        @test getmethod(algorithm) isa Unscented
        @test getL(getmethod(algorithm)) == 3
        @test getα(getmethod(algorithm)) == 1.0e-3
        @test getβ(getmethod(algorithm)) == 2.0
        @test getκ(getmethod(algorithm)) == 0.0
        @test getλ(getmethod(algorithm)) == 3.0e-6 - 3
    end

    params = [1.0, 2.0, 3.0]
    model = compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow(); permute = false),)), params)
    inputs = [
        MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])),
        MvNormalMeanPrecision([3.0, -1.5], diagm([1.0, 0.5])),
        MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 0.5])),
    ]

    @testset "Unscented() takes the dimension from the input" begin
        with_dim = FlowApproximation(model; method = Unscented(2))
        without_dim = FlowApproximation(model; method = Unscented())
        for m in inputs
            @test call_message_update_rule(Flow, :out; m = (in = m,), algorithm = without_dim) ≈ call_message_update_rule(Flow, :out; m = (in = m,), algorithm = with_dim)
            @test call_message_update_rule(Flow, :in; m = (out = m,), algorithm = without_dim) ≈ call_message_update_rule(Flow, :in; m = (out = m,), algorithm = with_dim)
        end
    end

    @testset "Unscented(dim) must match the flow's dimension" begin
        algorithm = FlowApproximation(model; method = Unscented(3))
        for m in inputs
            @test_throws DimensionMismatch call_message_update_rule(Flow, :out; m = (in = m,), algorithm)
            @test_throws DimensionMismatch call_message_update_rule(Flow, :in; m = (out = m,), algorithm)
        end
    end

    @testset "No rule under the default algorithm" begin
        m = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0]))
        @test_throws MessagePassingRulesBase.RuleNotFoundError call_message_update_rule(Flow, :out; m = (in = m,))
        @test_throws MessagePassingRulesBase.RuleNotFoundError call_message_update_rule(Flow, :in; m = (out = m,))
        @test_throws MessagePassingRulesBase.RuleNotFoundError call_message_update_rule(Flow, :out; m = (in = m,), algorithm = DefaultAlgorithm())
    end
end
