# Mixture: the cases are hand-derived. Its rules read their inputs' log scales, given to each call
# as `logscale`, keyed like the messages.

@testitem "rules:Mixture:inputs-out-switch" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using LogExpFunctions: logsumexp, softmax

    π = [0.3, 0.7]
    inputs = (NormalMeanVariance(-2.0, 1.0), NormalMeanVariance(2.0, 1.0))
    component_logscales = (-0.3, 0.4)

    # Towards input 2: m_out itself, with log scale ls_out + ls_switch + log π₂.
    result = call_message_update_rule(Mixture, (:inputs, 2); m = (out = NormalMeanVariance(1.0, 2.0), switch = Categorical(π)), logscale = (out = -0.5, switch = 0.2))
    @test getresult(result) == NormalMeanVariance(1.0, 2.0)
    @test getlogscale(result) ≈ -0.5 + 0.2 + log(0.7)

    # Towards out: the components, weighted by softmax(ls_k + ls_switch + log π_k).
    result = call_message_update_rule(Mixture, :out; m = (switch = Categorical(π), inputs = inputs), logscale = (switch = 0.1, inputs = component_logscales))
    evidence = [component_logscales...] .+ 0.1 .+ log.(π)
    @test getresult(result) isa MixtureDistribution
    @test collect(BayesBase.components(getresult(result))) == collect(inputs)
    @test BayesBase.weights(getresult(result)) ≈ softmax(evidence)
    @test getlogscale(result) ≈ logsumexp(evidence)

    # Towards switch: component k's evidence is ls_out + ls_k + log ∫ N(x; 1, 1/2) N(x; μ_k, 1) dx,
    # the last being log N(1; μ_k, 3/2), whichever product strategy the algorithm names.
    evidence = [0.2 + component_logscales[k] + logpdf(Normal(mean(inputs[k]), sqrt(1.5)), 1.0) for k in 1:2]
    for algorithm in (MixtureBP(), MixtureBP(prod = ClosedProd()))
        result = call_message_update_rule(Mixture, :switch; m = (out = NormalMeanVariance(1.0, 0.5), inputs = inputs), logscale = (out = 0.2, inputs = component_logscales), algorithm)
        @test probvec(getresult(result)) ≈ softmax(evidence)
        @test getlogscale(result) ≈ logsumexp(evidence)
    end
end

@testitem "rules:Mixture:needs-logscales" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    m = (out = NormalMeanVariance(1.0, 2.0), switch = Categorical([0.5, 0.5]))
    # Without log scales, the rule says it reads them.
    @test_throws "reads the log scales of its inbound messages" call_message_update_rule(Mixture, (:inputs, 1); m)
    # With one undefined, the error names its reason.
    @test_throws UndefinedLogScaleError call_message_update_rule(Mixture, (:inputs, 1); m, logscale = (out = UndefinedLogScale(:initial), switch = 0.0))
    @test_throws "an initial one" call_message_update_rule(Mixture, (:inputs, 1); m, logscale = (out = UndefinedLogScale(:initial), switch = 0.0))
    # No average energy: a free energy with a Mixture is an error.
    @test_throws MessagePassingRulesBase.RuleNotFoundError getresult(call_average_energy(Mixture; m = (out = NormalMeanVariance(0.0, 1.0), switch = Categorical([0.5, 0.5]), inputs = (NormalMeanVariance(0.0, 1.0), NormalMeanVariance(1.0, 1.0)))))
end
