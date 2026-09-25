@testmodule VerifiedRules begin
    using MessagePassingRulesBase, Distributions, BayesBase

    struct Gauss
        μ::Float64
        σ::Float64
    end
    BayesBase.logpdf(d::Gauss, x) = logpdf(Normal(d.μ, d.σ), x)
    @define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :σ])

    @define_message_update_rule(node = Gauss, target = :out, args = (m[:μ]::PointMass, m[:σ]::PointMass), body = (args) -> Normal(mean(args.m[:μ]), mean(args.m[:σ])))
    @define_message_update_rule(
        node = Gauss, target = :μ, args = (m[:out]::Normal, m[:σ]::PointMass),
        logscale = 0.0,
        body = (args) -> Normal(mean(args.m[:out]), sqrt(var(args.m[:out]) + mean(args.m[:σ])^2)),
    )
    @define_message_update_rule(
        node = Gauss, target = :μ, args = (q[:out]::Normal, q[:σ]::PointMass),
        body = (args) -> Normal(mean(args.q[:out]), mean(args.q[:σ])),
    )

    # Wrong on purpose: forgets the variance of the incoming message.
    struct WrongShape
        μ::Float64
        σ::Float64
    end
    BayesBase.logpdf(d::WrongShape, x) = logpdf(Normal(d.μ, d.σ), x)
    @define_factor_node(node = WrongShape, type = Stochastic, interfaces = [:out, :μ, :σ])
    @define_message_update_rule(
        node = WrongShape, target = :μ, args = (m[:out]::Normal, m[:σ]::PointMass),
        logscale = 0.0,
        body = (args) -> Normal(mean(args.m[:out]), mean(args.m[:σ])),
    )

    # Right shape, wrong normalisation.
    struct WrongScale
        μ::Float64
        σ::Float64
    end
    BayesBase.logpdf(d::WrongScale, x) = logpdf(Normal(d.μ, d.σ), x)
    @define_factor_node(node = WrongScale, type = Stochastic, interfaces = [:out, :μ, :σ])
    @define_message_update_rule(
        node = WrongScale, target = :μ, args = (m[:out]::Normal, m[:σ]::PointMass),
        logscale = 1.0,
        body = (args) -> Normal(mean(args.m[:out]), sqrt(var(args.m[:out]) + mean(args.m[:σ])^2)),
    )

    # Two continuous inputs, integrated in two dimensions.
    struct SumOfTwo
        a::Float64
        b::Float64
    end
    BayesBase.logpdf(d::SumOfTwo, x) = logpdf(Normal(d.a + d.b, 1.0), x)
    @define_factor_node(node = SumOfTwo, type = Stochastic, interfaces = [:out, :a, :b])
    @define_message_update_rule(
        node = SumOfTwo, target = :out, args = (m[:a]::Normal, m[:b]::Normal),
        logscale = 0.0,
        body = (args) -> Normal(mean(args.m[:a]) + mean(args.m[:b]), sqrt(1 + var(args.m[:a]) + var(args.m[:b]))),
    )

    # A default rule reached under an extension runs with `DefaultAlgorithm()`.
    struct Extended <: DefaultAlgorithmExtension end
    only_default(::DefaultAlgorithm, x) = x
    struct Shifted
        μ::Float64
        σ::Float64
    end
    BayesBase.logpdf(d::Shifted, x) = logpdf(Normal(d.μ, d.σ), x)
    @define_factor_node(node = Shifted, type = Stochastic, interfaces = [:out, :μ, :σ])
    @define_message_update_rule(
        node = Shifted, target = :out, args = (m[:μ]::PointMass, m[:σ]::PointMass),
        body = (algo, args) -> Normal(only_default(algo, mean(args.m[:μ])), mean(args.m[:σ])),
    )

    # A discrete input, enumerated.
    @define_factor_node(node = Bernoulli, type = Stochastic, interfaces = [:out, :p])
    @define_message_update_rule(
        node = Bernoulli, target = :p, args = (q[:out]::Bernoulli,),
        body = (args) -> Beta(1 + mean(args.q[:out]), 2 - mean(args.q[:out])),
    )
end

@testitem "verification:correct-rules" tags = [:testutils] setup = [VerifiedRules, Recording] begin
    using MessagePassingRulesTestUtils, Distributions, BayesBase
    V = VerifiedRules
    set = Recording.recorded() do
        @verify_message_update_rule(node = V.Gauss, target = :out, m = (μ = PointMass(1.0), σ = PointMass(2.0)))
        @verify_message_update_rule(node = V.Gauss, target = :μ, m = (out = Normal(0.5, 1.5), σ = PointMass(2.0)))
        @verify_message_update_rule(node = V.Gauss, target = :μ, q = (out = Normal(0.5, 1.5), σ = PointMass(2.0)))
        @verify_message_update_rule(node = V.SumOfTwo, target = :out, m = (a = Normal(1.0, 0.5), b = Normal(-2.0, 0.8)))
        @verify_message_update_rule(node = Bernoulli, target = :p, q = (out = Bernoulli(0.3),))
        @verify_message_update_rule(node = V.Shifted, target = :out, algorithm = V.Extended(), m = (μ = PointMass(1.0), σ = PointMass(2.0)))
    end
    @test isempty(Recording.failures(set))
    # Shape for all six, scale for the two rules that declare one.
    @test Recording.passes(set) == 6 + 2
end

@testitem "verification:negative-controls" tags = [:testutils] setup = [VerifiedRules, Recording] begin
    using MessagePassingRulesTestUtils, Distributions, BayesBase
    V = VerifiedRules

    shape = Recording.recorded() do
        @verify_message_update_rule(node = V.WrongShape, target = :μ, m = (out = Normal(0.5, 1.5), σ = PointMass(2.0)))
    end
    @test length(Recording.failures(shape)) == 1
    @test contains(Recording.failure_text(shape), "log-ratio to the node definition varies")

    scale = Recording.recorded() do
        @verify_message_update_rule(node = V.WrongScale, target = :μ, m = (out = Normal(0.5, 1.5), σ = PointMass(2.0)))
    end
    @test length(Recording.failures(scale)) == 1
    @test contains(Recording.failure_text(scale), "implies log scale")
    @test Recording.passes(scale) == 1                       # its shape is right

    @test_throws ArgumentError verify_message_update_rule(V.Gauss, :μ; m = (out = Normal(0.0, 1.0),))
    @test_throws ArgumentError verify_message_update_rule(V.Gauss, :μ; m = (out = Normal(0.0, 1.0),), q = (σ = PointMass(1.0),))
end
