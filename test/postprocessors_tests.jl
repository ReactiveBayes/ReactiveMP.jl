# The damping postprocessor mixes each message of a stream with the stream's previous one, in
# natural parameters, through the ExponentialFamily extension; anything else passes.

@testitem "DampingPostprocessor: the damping factor is in [0, 1)" tags = [:engine] begin
    import ReactiveMP: DampingPostprocessor
    @test DampingPostprocessor(0.5).β === 0.5
    @test DampingPostprocessor(0).β === 0
    @test_throws ArgumentError DampingPostprocessor(1.0)
    @test_throws ArgumentError DampingPostprocessor(-0.1)
end

@testitem "damp: natural parameters of one family" tags = [:engine] begin
    import ReactiveMP: damp
    using BayesBase, ExponentialFamily, Distributions

    # N(0, 1) and N(2, 4) have natural parameters (0, -1/2) and (1/2, -1/8); half of each is
    # (1/4, -5/16), the normal with variance 8/5 and mean 2/5.
    damped = damp(0.5, NormalMeanVariance(0.0, 1.0), NormalMeanVariance(2.0, 4.0))
    @test mean(damped) ≈ 0.4 && var(damped) ≈ 1.6
    # Across parameterisations of one family, and with all of the new one.
    @test mean(damp(0.5, NormalMeanPrecision(0.0, 1.0), NormalWeightedMeanPrecision(0.5, 0.25))) ≈ 0.4
    @test mean(damp(0.0, NormalMeanVariance(0.0, 1.0), NormalMeanVariance(2.0, 4.0))) ≈ 2.0
    # Gamma(2, rate 1) and Gamma(4, rate 3): natural parameters (1, -1) and (3, -3), half (2, -2).
    gamma = damp(0.5, GammaShapeRate(2.0, 1.0), GammaShapeRate(4.0, 3.0))
    @test mean(gamma) ≈ 3.0 / 2.0 && var(gamma) ≈ 3.0 / 4.0
    # Different families, a point mass, or something that is no distribution: the new one.
    new = NormalMeanVariance(2.0, 4.0)
    @test damp(0.5, GammaShapeRate(2.0, 1.0), new) === new
    @test damp(0.5, PointMass(1.0), new) === new
    @test damp(0.5, NormalMeanVariance(0.0, 1.0), PointMass(1.0)) === PointMass(1.0)
    @test damp(0.5, missing, new) === new
    @test damp(0.5, new, missing) === missing
end

@testitem "DampingPostprocessor: a stream of messages" tags = [:engine] begin
    import ReactiveMP: DampingPostprocessor, postprocess_stream_of_outbound_messages, postprocess_stream_of_marginals,
        postprocess_stream_of_scores, AbstractMessage, AnnotationDict
    import MessagePassingRulesBase: UndefinedLogScale
    using BayesBase, ExponentialFamily, Rocket

    damping = DampingPostprocessor(0.5)
    through(values) = (received = []; subscribe!(postprocess_stream_of_outbound_messages(damping, from(values)), (v) -> push!(received, v)); received)

    # The first message passes; each next one is mixed with the previous damped one.
    first, second, third = NormalMeanVariance(0.0, 1.0), NormalMeanVariance(2.0, 4.0), NormalMeanVariance(1.0, 1.0)
    received = through(AbstractMessage[Message(first, false, false), Message(second, false, true), Message(third, false, false)])
    @test getdata(received[1]) === first
    @test mean(getdata(received[2])) ≈ 0.4 && var(getdata(received[2])) ≈ 1.6
    expected = ReactiveMP.damp(0.5, getdata(received[2]), third)
    @test mean(getdata(received[3])) ≈ mean(expected) && var(getdata(received[3])) ≈ var(expected)
    # The flags of the new message are kept; a clamped one is never damped.
    @test is_initial(received[2]) && !is_initial(received[3])
    @test getdata(through(AbstractMessage[Message(first, false, false), Message(second, true, false)])[2]) === second

    # A tracked log scale is undefined once the message is damped; an untracked one stays so.
    tracked = through(AbstractMessage[Message(first, false, false, AnnotationDict(), 0.0), Message(second, false, false, AnnotationDict(), -1.0)])
    @test getlogscale(tracked[1]) === 0.0 && getlogscale(tracked[2]) isa UndefinedLogScale && getlogscale(tracked[2]).cause === :damping
    @test through(AbstractMessage[Message(first, false, false), Message(second, false, false)])[2].logscale === nothing

    # An equality chain's triggers, not messages, pass as they are; marginals and scores are left alone.
    @test through([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]
    stream = of(1.0)
    @test postprocess_stream_of_marginals(damping, stream) === stream && postprocess_stream_of_scores(damping, stream) === stream
end

@testitem "DampingPostprocessor: damped coordinate ascent reaches the same fixed point" tags = [:engine] setup = [EngineHarness] begin
    # μ ~ N(0, 100), τ ~ Gamma(1, 1), y[i] ~ N(μ, 1/τ) under q(μ)q(τ), with and without damping.
    using ExponentialFamily, Distributions, StandardMessagePassingRules
    import ReactiveMP: DampingPostprocessor
    H = EngineHarness

    Y = [1.2, 0.7, 2.1, 1.6, 0.9, 1.4]
    function model()
        graph = H.Graph()
        μ = H.random!(graph)
        μ_prior = [(:out, μ), (:μ, H.constant!(graph, 0.0)), (:τ, H.constant!(graph, 0.01))]
        τ = H.random!(graph)
        τ_prior = [(:out, τ), (:α, H.constant!(graph, 1.0)), (:β, H.constant!(graph, 1.0))]
        y = [H.data!(graph) for _ in Y]
        H.node!(graph, NormalMeanPrecision, μ_prior; factorisation = H.meanfield_factorisation(μ_prior))
        H.node!(graph, GammaShapeRate, τ_prior; factorisation = H.meanfield_factorisation(τ_prior))
        for i in eachindex(Y)
            interfaces = [(:out, y[i]), (:μ, μ), (:τ, τ)]
            H.node!(graph, NormalMeanPrecision, interfaces; factorisation = H.meanfield_factorisation(interfaces))
        end
        return graph, μ, τ, y
    end
    function posteriors(postprocessor, iterations)
        graph, μ, τ, y = model()
        result = H.run(
            graph; data = [y => Y], iterations, posteriors = [:μ => μ, :τ => τ],
            initial_marginals = [τ => GammaShapeRate(1.0, 1.0)], free_energy = false, postprocessor,
        )
        return result.posteriors["μ"], result.posteriors["τ"]
    end

    qμ, qτ = posteriors(nothing, 100)
    dμ, dτ = posteriors(DampingPostprocessor(0.5), 100)
    @test mean(dμ) ≈ mean(qμ) && var(dμ) ≈ var(qμ) && mean(dτ) ≈ mean(qτ)
    # Damped, it gets there more slowly: after two iterations it is further from the fixed point.
    early, _ = posteriors(nothing, 2)
    damped_early, _ = posteriors(DampingPostprocessor(0.5), 2)
    @test abs(mean(damped_early) - mean(qμ)) > abs(mean(early) - mean(qμ))
end
