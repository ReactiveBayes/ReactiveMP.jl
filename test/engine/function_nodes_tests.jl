# A node whose functional form is a function, `+`, runs through the engine like any other: the
# model x1 ~ N(0, 1), x2 ~ N(1, 2), s = x1 + x2, y ~ N(s, 1/2) with y = 3 observed is a tree, so
# belief propagation is exact and the free energy is -log p(y).

@testitem "engine:function-node" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, Distributions, BayesBase
    H = EngineHarness

    graph = H.Graph()
    y = H.data!(graph)
    x1, x2, s = H.random!(graph), H.random!(graph), H.random!(graph)
    H.node!(graph, NormalMeanVariance, [(:out, x1), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, NormalMeanVariance, [(:out, x2), (:μ, H.constant!(graph, 1.0)), (:v, H.constant!(graph, 2.0))])
    H.node!(graph, +, [(:out, s), (:in1, x1), (:in2, x2)])
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, s), (:v, H.constant!(graph, 0.5))])

    trajectory = H.run(graph; id = "function-node", data = [y => 3.0], iterations = 1, posteriors = [:x1 => x1, :x2 => x2, :s => s])
    q = trajectory.posteriors
    # y - E[y] = 2 is shared out with the gain Cov(x, y) / Var(y) = [1, 2] / 3.5.
    @test all(isapprox.(mean_var(q["x1"]), (2 / 3.5, 1 - 1 / 3.5)))
    @test all(isapprox.(mean_var(q["x2"]), (1 + 4 / 3.5, 2 - 4 / 3.5)))
    # s ~ N(1, 3) a priori; with y it is N(19/7, 3/7).
    @test all(isapprox.(mean_var(q["s"]), (19 / 7, 3 / 7)))
    @test only(trajectory.free_energy) ≈ -logpdf(Normal(1.0, sqrt(3.5)), 3.0)
end
