# Case (a) of Phase 4.5: belief propagation through NormalMeanVariance, compared call by call
# with the trajectories v6 recorded through RxInfer (`compat/v6-comparison/fixtures/engine`).

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
