# Mixture: the cases are hand-derived. Its rules read their inputs' log
# scales, which the tables cannot supply, so they are called directly with `ann`.

@testmodule MixtureInputs begin
    using MessagePassingRulesBase, BayesBase
    using MessagePassingRulesBase: AnnotationStore, RuleAnnotations, RuleContext, annotate!

    # An incoming message's annotations, carrying a log scale.
    logscaled(ls) = (store = AnnotationStore(); annotate!(store, :logscale, ls); store)
    annotations(; m) = RuleAnnotations(; m, out = AnnotationStore())
    # The engine's product service: the product with its own log scale.
    product(l, r) = (d = prod(GenericProd(), l, r); (d, compute_logscale(d, l, r)))
    const CTX = RuleContext(product = product)
end

@testitem "rules:Mixture:inputs-out-switch" tags = [:rules] setup = [MixtureInputs] begin
    using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using MessagePassingRulesBase: getannotation
    using LogExpFunctions: logsumexp, softmax
    M = MixtureInputs

    π = [0.3, 0.7]
    inputs = (NormalMeanVariance(-2.0, 1.0), NormalMeanVariance(2.0, 1.0))
    component_logscales = (-0.3, 0.4)

    # Towards input 2: m_out itself, with log scale ls_out + ls_switch + log π₂.
    ann = M.annotations(m = (out = M.logscaled(-0.5), switch = M.logscaled(0.2)))
    message = call_message_update_rule(Mixture, (:inputs, 2); m = (out = NormalMeanVariance(1.0, 2.0), switch = Categorical(π)), ann)
    @test message == NormalMeanVariance(1.0, 2.0)
    @test getannotation(ann, :logscale) ≈ -0.5 + 0.2 + log(0.7)

    # Towards out: the components, weighted by softmax(ls_k + ls_switch + log π_k).
    ann = M.annotations(m = (switch = M.logscaled(0.1), inputs = map(M.logscaled, component_logscales)))
    message = call_message_update_rule(Mixture, :out; m = (switch = Categorical(π), inputs = inputs), ann)
    evidence = [component_logscales...] .+ 0.1 .+ log.(π)
    @test message isa MixtureDistribution
    @test collect(BayesBase.components(message)) == collect(inputs)
    @test BayesBase.weights(message) ≈ softmax(evidence)
    @test getannotation(ann, :logscale) ≈ logsumexp(evidence)

    # Towards switch: component k's evidence is ls_out + ls_k + log ∫ N(x; 1, 1/2) N(x; μ_k, 1) dx,
    # the last being log N(1; μ_k, 3/2).
    ann = M.annotations(m = (out = M.logscaled(0.2), inputs = map(M.logscaled, component_logscales)))
    message = call_message_update_rule(Mixture, :switch; m = (out = NormalMeanVariance(1.0, 0.5), inputs = inputs), ann, ctx = M.CTX)
    evidence = [0.2 + component_logscales[k] + logpdf(Normal(mean(inputs[k]), sqrt(1.5)), 1.0) for k in 1:2]
    @test probvec(message) ≈ softmax(evidence)
    @test getannotation(ann, :logscale) ≈ logsumexp(evidence)
end

@testitem "rules:Mixture:needs-logscales" tags = [:rules] setup = [MixtureInputs] begin
    using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using MessagePassingRulesBase: AnnotationStore
    M = MixtureInputs

    # A message without a log scale: the rule names the annotations it needs.
    ann = M.annotations(m = (out = AnnotationStore(), switch = M.logscaled(0.0)))
    @test_throws r"LogScaleAnnotations" call_message_update_rule(Mixture, (:inputs, 1); m = (out = NormalMeanVariance(1.0, 2.0), switch = Categorical([0.5, 0.5])), ann)
    # No average energy: a free energy with a Mixture is an error.
    @test_throws MessagePassingRulesBase.RuleNotFoundError call_average_energy(Mixture; m = (out = NormalMeanVariance(0.0, 1.0), switch = Categorical([0.5, 0.5]), inputs = (NormalMeanVariance(0.0, 1.0), NormalMeanVariance(1.0, 1.0))))
end
