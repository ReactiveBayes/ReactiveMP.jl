# A single-input deterministic node's clusters are the variables' own marginals, and the
# marginal of `in` is the product of its incoming messages, `m_in` and the message towards `in`.
# These cases check that product against the expected marginals.

@testitem "rules:Flow:marginals" tags = [:rules] begin
    using FlowMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, MessagePassingRulesApproximations, BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    params = [1.0, 2.0, 3.0]
    model = FlowModel(2, (AdditiveCouplingLayer(PlanarFlow(); permute = false),))
    compiled_model = compile(model, params)

    # The same normal, `N([-5.0, -2.5], diagm([1.0, 2.0]))`, in each of the three forms.
    mc = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0]))
    mp = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1 / 2.0]))
    mw = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1 / 2.0]))

    # Nine cases, as `(m_out, m_in)`.
    pairs = [(mc, mc), (mp, mp), (mw, mw), (mc, mp), (mp, mc), (mw, mp), (mp, mw), (mc, mw), (mw, mc)]

    function check_marginal(algorithm, expected, atol)
        for (m_out, m_in) in pairs
            marginal = prod(GenericProd(), m_in, call_message_update_rule(Flow, :in; m = (out = m_out,), algorithm))
            @test marginal isa MvNormalWeightedMeanPrecision
            @test isapprox(weightedmean(marginal), weightedmean(expected); atol)
            @test isapprox(precision(marginal), precision(expected); atol)
            @test isapprox(mean(marginal), mean(expected); atol)
            @test isapprox(cov(marginal), cov(expected); atol)
        end
    end

    @testset ":in (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily) (Linearization)" begin
        expected = MvNormalWeightedMeanPrecision(
            [-10.75002245135493, -2.0000174620747515],
            [2.5000066522408155 0.5000033261093448; 0.5000033261093448 1.0],
        )
        check_marginal(FlowApproximation(compiled_model), expected, 1.0e-6)
    end

    @testset ":in (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily) (Unscented)" begin
        expected = MvNormalWeightedMeanPrecision(
            [-10.750029103730993, -2.0000241143779007],
            [2.5000066521881763 0.5000033260385859; 0.5000033260385859 0.9999999999115444],
        )
        check_marginal(FlowApproximation(compiled_model; method = Unscented(2)), expected, 1.0e-9)
    end
end
