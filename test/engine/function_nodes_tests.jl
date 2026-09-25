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

# `CVIProjection`'s message towards `out` is a `DivisionOf`, the projected marginal of `out` over
# the message on its edge: the engine's product at the variable, not the rule, divides it out.
# Through a linear function the posterior is exact: x ~ N(0.5, 1), z = x + 1, y ~ N(z, 0.1)
# observed at 2 give x a posterior of precision 11 and mean 10.5 / 11.
@testitem "engine:function-node:cvi-projection" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, ExponentialFamilyProjection, StandardMessagePassingRules, DeltaMessagePassingRules, BayesBase
    import Random
    H = EngineHarness
    Random.seed!(42)
    shift(x) = x + 1

    graph = H.Graph()
    x, z = H.random!(graph), H.random!(graph)
    y = H.data!(graph)
    H.node!(graph, NormalMeanVariance, [(:out, x), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, DeltaFn{typeof(shift)}, [(:out, z), ((:in, 1), x)]; algorithm = DeltaApproximation(method = CVIProjection(outsamples = 2000)), nodefn = shift)
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, z), (:v, H.constant!(graph, 0.1))])

    trajectory = H.run(
        graph; id = "cvi_projection", data = [y => 2.0], iterations = 10, free_energy = false,
        posteriors = [:z => z, :x => x], initial_marginals = [z => NormalMeanVariance(1.0, 1.0)],
    )
    q_x, q_z = trajectory.posteriors["x"], trajectory.posteriors["z"]
    # The marginal of `z` is the projection itself, a normal, not a lazy quotient: the product
    # at the variable divided the message on its edge back out.
    @test q_z isa UnivariateNormalDistributionsFamily
    @test mean(q_x) ≈ 10.5 / 11 atol = 0.1
    @test var(q_x) ≈ 1 / 11 atol = 0.05
    # The projection of `out` is only as close as sampling and ExponentialFamilyProjection make
    # it: measured, a mean of 1.86 and a variance of 0.22.
    @test mean(q_z) ≈ 10.5 / 11 + 1 atol = 0.25
    @test 0.0 < var(q_z) < 0.5
end
