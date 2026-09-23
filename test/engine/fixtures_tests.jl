# Cases (a) and (b) of Phase 4.5, belief propagation and variational message passing, compared
# call by call with the trajectories v6 recorded through RxInfer
# (`compat/v6-comparison/fixtures/engine`).
#
# The case (b) graphs are built in GraphPPL's order, which is the order RxInfer activates them
# in: variables as the model statements create them, each constant after the random variable of
# its statement, and data where it is first used. Posteriors are subscribed in the order
# RxInfer's `returnvars` `Dict` iterates, which is `μ, τ, x`.

@testitem "engine:fixture:bp_iid" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    import ReactiveMP: LogScaleAnnotations
    H = EngineHarness

    Y = [1.2, 0.7, 2.1, 1.6, 0.9, 1.4]
    graph = H.Graph()
    y = [H.data!(graph) for _ in Y]
    prior_v = H.data!(graph)
    v = H.data!(graph)
    x = H.random!(graph)
    zero_mean = H.constant!(graph, 0.0)
    H.node!(graph, NormalMeanVariance, [(:out, x), (:μ, zero_mean), (:v, prior_v)])
    for i in eachindex(Y)
        H.node!(graph, NormalMeanVariance, [(:out, y[i]), (:μ, x), (:v, v)])
    end

    trajectory = H.run(
        graph; id = "bp_iid", data = [y => Y, prior_v => 10.0, v => 1.0], iterations = 2,
        posteriors = [:x => x], annotations = (LogScaleAnnotations(),),
    )
    @test compare_engine_trajectory(trajectory, H.fixture("bp_iid"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:bp_iid_missing" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    Y = [1.2, missing, 2.1, 1.6, 0.9, 1.4]
    graph = H.Graph()
    y = [H.data!(graph) for _ in Y]
    prior_v = H.data!(graph)
    v = H.data!(graph)
    x = H.random!(graph)
    zero_mean = H.constant!(graph, 0.0)
    H.node!(graph, NormalMeanVariance, [(:out, x), (:μ, zero_mean), (:v, prior_v)])
    for i in eachindex(Y)
        H.node!(graph, NormalMeanVariance, [(:out, y[i]), (:μ, x), (:v, v)])
    end

    trajectory = H.run(
        graph; id = "bp_iid_missing", data = [y => Y, prior_v => 10.0, v => 1.0], iterations = 2,
        posteriors = [:x => x], predictions = y, free_energy = false,
    )
    @test compare_engine_trajectory(trajectory, H.fixture("bp_iid_missing"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:bp_chain" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    Y = [1.2, 0.7, 2.1, 1.6, 0.9, 1.4]
    graph = H.Graph()
    y = [H.data!(graph) for _ in Y]
    prior_v = H.data!(graph)
    v = H.data!(graph)
    x = [H.random!(graph) for _ in Y]
    zero_mean = H.constant!(graph, 0.0)
    H.node!(graph, NormalMeanVariance, [(:out, x[1]), (:μ, zero_mean), (:v, prior_v)])
    H.node!(graph, NormalMeanVariance, [(:out, y[1]), (:μ, x[1]), (:v, v)])
    for i in 2:length(Y)
        H.node!(graph, NormalMeanVariance, [(:out, x[i]), (:μ, x[i - 1]), (:v, v)])
        H.node!(graph, NormalMeanVariance, [(:out, y[i]), (:μ, x[i]), (:v, v)])
    end

    trajectory = H.run(
        graph; id = "bp_chain", data = [y => Y, prior_v => 10.0, v => 1.0], iterations = 2,
        posteriors = [:x => x],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("bp_chain"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:vmp_meanfield" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    Y = [1.2, 0.7, 2.1, 1.6, 0.9, 1.4]
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

    trajectory = H.run(
        graph; id = "vmp_meanfield", data = [y => Y], iterations = 5,
        posteriors = [:μ => μ, :τ => τ], initial_marginals = [τ => GammaShapeRate(1.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("vmp_meanfield"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:vmp_structured" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    Y = [1.2, 0.7, 2.1, 1.6, 0.9, 1.4]
    graph = H.Graph()
    μ = H.random!(graph)
    μ_prior = [(:out, μ), (:μ, H.constant!(graph, 0.0)), (:τ, H.constant!(graph, 0.01))]
    τ = H.random!(graph)
    τ_prior = [(:out, τ), (:α, H.constant!(graph, 1.0)), (:β, H.constant!(graph, 1.0))]
    x, y, observations = [], [], []
    for _ in Y
        push!(x, H.random!(graph))
        push!(y, H.data!(graph))
        push!(observations, [(:out, y[end]), (:μ, x[end]), (:v, H.constant!(graph, 0.5))])
    end
    H.node!(graph, NormalMeanPrecision, μ_prior)
    H.node!(graph, GammaShapeRate, τ_prior)
    for i in eachindex(Y)
        # q(x, μ)q(τ)
        H.node!(graph, NormalMeanPrecision, [(:out, x[i]), (:μ, μ), (:τ, τ)]; factorisation = ((:out, :μ), (:τ,)))
        H.node!(graph, NormalMeanVariance, observations[i])
    end

    trajectory = H.run(
        graph; id = "vmp_structured", data = [y => Y], iterations = 5,
        posteriors = [:μ => μ, :τ => τ, :x => x],
        initial_marginals = [τ => GammaShapeRate(1.0, 1.0), μ => NormalMeanPrecision(0.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("vmp_structured"); atol = 1.0e-9) === :agree
end
