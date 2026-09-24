# `CVIProjection`, from v6's `test/ext/ReactiveMPProjectionExt/`. The rules draw from the
# context's generator, a StableRNG here; the checks are v6's, statistical and loose. v6's JET
# checks and its tests of `optimize_parameters` and `create_density_function`, helpers no rule
# called, are not ported, nor is its benchmark of the two sampling strategies.

@testmodule CVIContext begin
    using MessagePassingRulesBase: MessagePassingRulesBase, Target, RuleContext
    using StableRNGs: StableRNG

    struct FunctionNode{F}
        f::F
    end
    MessagePassingRulesBase.getnodefn(node::FunctionNode, ::Target{:out}) = node.f

    context(f; seed = 42) = RuleContext(node = FunctionNode(f), rng = StableRNG(seed))
    diagonal(v) = [i == j ? float(v[i]) : 0.0 for i in eachindex(v), j in eachindex(v)]
    identity_matrix(n) = diagonal(ones(n))
end

@testitem "cvi:DivisionOf" tags = [:rules] begin
    using DeltaMessagePassingRules, ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(DeltaMessagePassingRules, :DeltaMessagePassingRulesProjectionExt)
    @test ext !== nothing
    d1, d2 = NormalMeanVariance(0, 1), NormalMeanVariance(1, 2)
    @test d1 ≈ prod(GenericProd(), ext.DivisionOf(d1, d2), d2)
    @test d1 ≈ prod(GenericProd(), d2, ext.DivisionOf(d1, d2))
    @test ext.DivisionOf(d1, d2) == prod(GenericProd(), ext.DivisionOf(d1, d2), missing)
    @test ext.DivisionOf(d1, d2) == prod(GenericProd(), missing, ext.DivisionOf(d1, d2))
end

@testitem "cvi:algorithm" tags = [:rules] begin
    using DeltaMessagePassingRules, ExponentialFamily, ExponentialFamilyProjection, MessagePassingRulesBase
    import DeltaMessagePassingRules: get_kth_in_form, delta_method_hint

    # The extension opts the method in; without it, the error would add this hint.
    @test DeltaApproximation(method = CVIProjection()).method isa CVIProjection
    @test contains(delta_method_hint(CVIProjection()), "using ExponentialFamilyProjection")
    @test_logs (:warn, r"does not use a known inverse") DeltaApproximation(method = CVIProjection(), inverse = identity)

    form1, form2 = ProjectedTo(NormalMeanVariance), ProjectedTo(MvNormalMeanScalePrecision, 2)
    with_forms = CVIProjection(in_prjparams = (in_1 = form1, in_2 = form2))
    @test get_kth_in_form(with_forms, 1) === form1 && get_kth_in_form(with_forms, 2) === form2
    @test get_kth_in_form(with_forms, 3) === nothing
    @test get_kth_in_form(CVIProjection(), 1) === nothing

    # The joint rule replaces the proposal, so it is impure.
    spec = which_marginal_update_rule(DeltaFn, (:in,); m = (out = 1.0, in = (1.0, 2.0)), algorithm = DeltaApproximation(method = CVIProjection()))
    @test !spec.pure
end

@testitem "cvi:in" tags = [:rules] begin
    using DeltaMessagePassingRules, ExponentialFamily, ExponentialFamilyProjection, BayesBase, MessagePassingRulesBase

    algorithm = DeltaApproximation(method = CVIProjection())
    m_in = NormalMeanVariance(0, 1)
    q_ins = FactorizedJoint((prod(GenericProd(), m_in, NormalMeanVariance(0, 1)),))
    message = call_message_update_rule(DeltaFn, (:in, 1); m = (in = (m_in,),), clusters = ((:in,) => q_ins,), algorithm)
    @test prod(GenericProd(), message, m_in) ≈ component(q_ins, 1)
end

@testitem "cvi:out: the identity, for several families" tags = [:rules] setup = [CVIContext] begin
    using DeltaMessagePassingRules, ExponentialFamily, ExponentialFamilyProjection, BayesBase, Distributions, MessagePassingRulesBase
    using .CVIContext: context

    algorithm = DeltaApproximation(method = CVIProjection())
    cases = [
        (NormalMeanVariance(2, 2), NormalMeanVariance(0, 1)), (Gamma(2, 1), Gamma(2, 2)), (Beta(5, 4), Beta(5, 5)),
        (Rayleigh(4), Rayleigh(5.2)), (Geometric(0.3), Geometric(0.9)), (LogNormal(0.2, 1.0), LogNormal(3.0, 2.1)),
        (Exponential(0.3), Exponential(4.3)),
    ]
    for (q_in, m_out) in cases
        q_out = q_in
        message = call_message_update_rule(DeltaFn, :out; m = (out = m_out,), q = (out = q_out,), clusters = ((:in,) => FactorizedJoint((q_in,)),), algorithm, ctx = context(identity))
        projected = project_to(ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...), x -> logpdf(message, x) + logpdf(m_out, x); initialpoint = q_out)
        @test mean(projected) ≈ mean(q_out) atol = 5.0e-1
        @test var(projected) ≈ var(q_out) atol = 2.0
        @test mode(projected) ≈ mode(q_out) atol = 5.0e-1
        q_out isa Union{Exponential, Beta, Gamma} && @test mean(log, projected) ≈ mean(log, q_out) atol = 5.0e-1
    end
end

# `f(x, y) = [x; y]` takes univariate and multivariate inputs to a multivariate output, whose
# mean is the inputs' means stacked.
@testitem "cvi:out: stacking inputs of any variate form" tags = [:rules] setup = [CVIContext] begin
    using DeltaMessagePassingRules, ExponentialFamily, ExponentialFamilyProjection, BayesBase, MessagePassingRulesBase
    using .CVIContext: context, diagonal, identity_matrix

    algorithm = DeltaApproximation(method = CVIProjection(outsamples = 1000))
    f(x, y) = [x; y]
    cases = [
        (FactorizedJoint((NormalMeanVariance(3.0, 1.0), MvNormalMeanCovariance([2.0, 5.2], 3 * identity_matrix(2)))), MvNormalMeanCovariance([2.0, 3.4, -1.0], diagonal([0.2, 0.01, 4.0]))),
        (FactorizedJoint((MvNormalMeanCovariance([0.3, 0.9], identity_matrix(2)), MvNormalMeanCovariance([2.0, 5.2], 3 * identity_matrix(2)))), MvNormalMeanCovariance([2.0, 3.4, -1.0, 5.0], diagonal([0.2, 2.0, 0.01, 4.0]))),
        (FactorizedJoint((MvNormalMeanCovariance([2.0, 3.0, 0.1, 0.9], identity_matrix(4)), MvNormalMeanCovariance([3.4, 7.6], 0.1 * identity_matrix(2)))), MvNormalMeanCovariance([2.0, 3.4, -1.0, 5.0, 3.0, -10.0], diagonal([0.2, 2.0, 0.01, 4.0, 1.0, 0.5]))),
    ]
    for (q_ins, m_out) in cases
        parts = components(q_ins)
        q_out = MvNormalMeanCovariance(mapreduce(mean, vcat, parts), diagonal(mapreduce(var, vcat, parts)))
        message = call_message_update_rule(DeltaFn, :out; m = (out = m_out,), q = (out = q_out,), clusters = ((:in,) => q_ins,), algorithm, ctx = context(f))
        projected = project_to(ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...), x -> logpdf(message, x) + logpdf(m_out, x); initialpoint = q_out)
        @test mean(projected) ≈ mean(q_out) rtol = 5.0e-1
        @test var(projected) ≈ var(q_out) rtol = 1.0
        @test mode(projected) ≈ mode(q_out) rtol = 5.0e-1
    end
end

@testitem "cvi:out: affine functions of a normal" tags = [:rules, :slow] setup = [CVIContext] begin
    using DeltaMessagePassingRules, ExponentialFamily, ExponentialFamilyProjection, BayesBase, MessagePassingRulesBase
    using .CVIContext: context

    algorithm = DeltaApproximation(method = CVIProjection(outsamples = 10000))
    for q_in in (NormalMeanVariance(0, 2), NormalMeanVariance(3, 4), NormalMeanVariance(5, 4)), a in (1, 1.1), b in -3:3,
            m_out in (NormalMeanVariance(0, 1), NormalMeanVariance(2, 3), NormalMeanVariance(4, 0.1))
        f = x -> a * x + b
        q_out = NormalMeanVariance(a * mean(q_in) + b, a^2 * var(q_in))
        message = call_message_update_rule(DeltaFn, :out; m = (out = m_out,), q = (out = q_out,), clusters = ((:in,) => FactorizedJoint((q_in,)),), algorithm, ctx = context(f))
        projected = project_to(ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...), x -> logpdf(message, x) + logpdf(m_out, x))
        @test mean(projected) ≈ mean(q_out) atol = 5.0e-1
        @test var(projected) ≈ var(q_out) atol = 7.0e-1
    end
end

@testitem "cvi:out: a shift of a multivariate normal" tags = [:rules] setup = [CVIContext] begin
    using DeltaMessagePassingRules, ExponentialFamily, ExponentialFamilyProjection, BayesBase, MessagePassingRulesBase
    using .CVIContext: context, identity_matrix

    algorithm = DeltaApproximation(method = CVIProjection())
    c = [0.2, -9.0, 3.0]
    q_in = MvNormalMeanCovariance([0.3, 0.7, 10.0], 0.1 * identity_matrix(3))
    m_out = MvNormalMeanCovariance(ones(3), 0.9 * identity_matrix(3))
    q_out = MvNormalMeanCovariance(mean(q_in) + c, cov(q_in))
    message = call_message_update_rule(DeltaFn, :out; m = (out = m_out,), q = (out = q_out,), clusters = ((:in,) => FactorizedJoint((q_in,)),), algorithm, ctx = context(x -> x + c))
    projected = project_to(ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...), x -> logpdf(message, x), m_out; initialpoint = q_out)
    @test mean(projected) ≈ mean(q_out) rtol = 5.0e-1
    @test var(projected) ≈ var(q_out) rtol = 5.0e-1
end

# Beta(a, b) through 1 - x is Beta(b, a); Gamma(a, θ) through cx is Gamma(a, cθ); Exp(λ)
# through √x is Rayleigh(1/√(2λ)), and through exp(-x) Beta(λ, 1).
@testitem "cvi:out: nonlinear functions of other families" tags = [:rules] setup = [CVIContext] begin
    using DeltaMessagePassingRules, ExponentialFamily, ExponentialFamilyProjection, BayesBase, Distributions, MessagePassingRulesBase
    using .CVIContext: context

    algorithm = DeltaApproximation(method = CVIProjection())
    cases = [
        (Beta(5, 2), Beta(20, 3), Beta(2, 5), x -> 1 - x),
        (Gamma(3, 4), Gamma(10, 7), Gamma(3, 4 * 0.1), x -> 0.1 * x),
        (Exponential(0.5), Exponential(3), Rayleigh(1 / sqrt(2 * 3)), x -> sqrt(x)),
        (Exponential(0.5), Exponential(30), Beta(0.5, 1), x -> exp(-x)),
    ]
    for (q_in, m_out, q_out, f) in cases
        message = call_message_update_rule(DeltaFn, :out; m = (out = m_out,), q = (out = q_out,), clusters = ((:in,) => FactorizedJoint((q_in,)),), algorithm, ctx = context(f))
        projected = project_to(ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...), x -> logpdf(message, x) + logpdf(m_out, x))
        @test mean(projected) ≈ mean(q_out) atol = 5.0e-1
        @test var(projected) ≈ var(q_out) atol = 1.0e-1
    end
end

@testitem "cvi:out: a family named for the message" tags = [:rules] setup = [CVIContext] begin
    using DeltaMessagePassingRules, ExponentialFamily, ExponentialFamilyProjection, BayesBase, MessagePassingRulesBase
    using .CVIContext: context, identity_matrix

    # exp of a standard normal, projected onto a LogNormal: E[exp(x)] = exp(1/2).
    algorithm = DeltaApproximation(method = CVIProjection(out_prjparams = ProjectedTo(LogNormal), outsamples = 1000))
    message = call_message_update_rule(DeltaFn, :out; m = (out = Gamma(2.0, 2.0),), q = (out = Gamma(2.0, 2.0),), clusters = ((:in,) => FactorizedJoint((NormalMeanVariance(0.0, 1.0),)),), algorithm, ctx = context(exp))
    @test message.numerator isa LogNormal
    @test mean(message.numerator) ≈ exp(1 / 2) rtol = 0.05

    # Each component squared, projected onto a scale-precision normal: E[x²] = μ² + σ².
    algorithm = DeltaApproximation(method = CVIProjection(out_prjparams = ProjectedTo(MvNormalMeanScalePrecision, 2), outsamples = 10000))
    Σ = [1.0 0.5; 0.5 1.0]
    m_out = q_out = MvNormalMeanCovariance(ones(2), identity_matrix(2))
    for μ in ([0.0, 0.0], [7.0, -3.0])
        message = call_message_update_rule(DeltaFn, :out; m = (out = m_out,), q = (out = q_out,), clusters = ((:in,) => FactorizedJoint((MvNormalMeanCovariance(μ, Σ),)),), algorithm, ctx = context(x -> x .^ 2))
        @test message.numerator isa MvNormalMeanScalePrecision
        @test mean(message.numerator) ≈ μ .^ 2 .+ [1.0, 1.0] rtol = 0.1
    end
end

@testitem "cvi:joint" tags = [:rules] setup = [CVIContext] begin
    using DeltaMessagePassingRules, ExponentialFamily, ExponentialFamilyProjection, BayesBase, Distributions, MessagePassingRulesBase
    using .CVIContext: context

    joint(algorithm, f, m_out, m_ins...) = call_marginal_update_rule(DeltaFn, (:in,); m = (out = m_out, in = m_ins), algorithm, ctx = context(f))

    # Through the identity, the joint of one input is its message times the message from `out`.
    algorithm = DeltaApproximation(method = CVIProjection())
    for m in (NormalMeanVariance(0, 1), Gamma(2, 2), Beta(1, 1), MvNormalMeanCovariance([0.5, 0.5]), MvNormalMeanCovariance([0.5, 0.5, -1.0]))
        q = joint(algorithm, identity, m, m)
        @test length(components(q)) == 1
        @test component(q, 1) ≈ prod(GenericProd(), m, m) atol = 1.0e-1
    end
    for (m_in, m_out) in ((Binomial(3, 0.9), Binomial(7, 0.4)), (Binomial(8, 0.9), Binomial(8, 0.9)), (Binomial(5, 0.01), Binomial(6, 0.98)))
        product = prod(PreserveTypeProd(ExponentialFamilyDistribution), m_in, m_out)
        grid = getsupport(product)
        @test mean(component(joint(algorithm, identity, m_out, m_in), 1)) ≈ sum(grid .* pdf(product, grid)) atol = 1.0e-1
    end
    categorical = DeltaApproximation(method = CVIProjection(in_prjparams = (in_1 = ExponentialFamilyProjection.ProjectionParameters(strategy = ExponentialFamilyProjection.ControlVariateStrategy(nsamples = 4_000)),)))
    for (m_in, m_out) in ((Categorical([1 / 4, 1 / 4, 1 / 2]), Categorical([1 / 2, 1 / 8, 3 / 8])), (Categorical([1 / 2, 1 / 8, 3 / 8]), Categorical([1 / 16, 13 / 16, 1 / 8])), (Categorical([1 / 8, 1 / 8, 1 / 8, 5 / 8]), Categorical([2 / 7, 2 / 7, 1 / 7, 2 / 7])))
        @test mean(component(joint(categorical, identity, m_out, m_in), 1)) ≈ mean(prod(GenericProd(), m_in, m_out)) atol = 2.0e-1
    end

    # Two inputs stacked, by full sampling and by the means: the output's message is exact here.
    f(x, y) = [x, y]
    for strategy in (FullSampling(10), MeanBased())
        q = joint(DeltaApproximation(method = CVIProjection(sampling_strategy = strategy)), f, MvNormalMeanCovariance(ones(2), [2.0 0.0; 0.0 2.0]), NormalMeanVariance(0, 1), NormalMeanVariance(1, 2))
        @test component(q, 1) ≈ NormalMeanVariance(1 / 3, 2 / 3) atol = 1.0e-1
        @test component(q, 2) ≈ NormalMeanVariance(1.0, 1.0) atol = 1.0e-1
    end

    # A family named for the second input only: the first keeps its own.
    partial = DeltaApproximation(method = CVIProjection(in_prjparams = (in_2 = ProjectedTo(MvNormalMeanScalePrecision, 2),), sampling_strategy = FullSampling(10)))
    q = joint(partial, (x, y) -> x .* y, MvNormalMeanCovariance([2.0, 3.0], [1.0 0.0; 0.0 1.0]), Gamma(2.0, 2.0), MvNormalMeanCovariance([1.0, 1.0], [2.0 0.0; 0.0 2.0]))
    @test component(q, 1) isa Gamma
    @test component(q, 2) isa MvNormalMeanScalePrecision
end

# The joint rule replaces the proposal with its result, and the next call samples the inputs from
# it; within a call the inputs are projected in turn, each against the others' latest
# projections. Repeated calls then bring the proposal closer to the posterior, by the KL
# divergence, for `x * y` observed at 2, whose posterior has two modes. v6 projected every input
# against the same samples of the previous proposal, and there the inputs flip between modes of
# opposite sign (`PHASES.md` § Phase 6, step 2).
@testitem "cvi:joint: the proposal converges" tags = [:rules] setup = [CVIContext] begin
    using DeltaMessagePassingRules, ExponentialFamily, ExponentialFamilyProjection, BayesBase, Distributions, MessagePassingRulesBase, StableRNGs
    using .CVIContext: context

    method = CVIProjection(sampling_strategy = FullSampling(500))
    algorithm = DeltaApproximation(method = method)
    f(x, y) = x * y
    m_out, m_x, m_y = NormalMeanVariance(2.0, 0.1), NormalMeanVariance(0.0, 2.0), NormalMeanVariance(0.0, 2.0)
    rng = StableRNG(123)
    function kl_from_posterior(q)
        samples = [(rand(rng, component(q, 1)), rand(rng, component(q, 2))) for _ in 1:1000]
        return mean(logpdf(component(q, 1), x) + logpdf(component(q, 2), y) - (logpdf(m_x, x) + logpdf(m_y, y) + logpdf(m_out, f(x, y))) for (x, y) in samples)
    end
    results = map(seed -> call_marginal_update_rule(DeltaFn, (:in,); m = (out = m_out, in = (m_x, m_y)), algorithm, ctx = context(f; seed)), 1:10)
    divergences = map(kl_from_posterior, results)
    @test first(divergences) > 2 * last(divergences)
    # It settles on one mode: both inputs of one sign, as `x * y = 2` requires.
    @test prod(mean, components(last(results))) > 0.5
    # And the proposal is the last result.
    @test method.proposal_distribution.distribution === last(results)
end
