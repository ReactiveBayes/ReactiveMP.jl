# A marginal rule that returns a `FactorizedCluster` says the cluster splits into independent
# blocks. Whoever reads the cluster, a message rule or the node's average energy, receives the
# blocks under their own keys, and its entropy is the sum over the blocks.

@testitem "engine:factorized-cluster:arguments" tags = [:engine] begin
    using BayesBase, Distributions, ExponentialFamily, MessagePassingRulesBase
    import ReactiveMP: rule_marginals, rule_arguments, getdata, getannotations, Marginal, AnnotationDict

    split = Marginal(FactorizedCluster((:out,) => PointMass(1.0), (:μ,) => Normal(0.0, 2.0)), false, false)
    v = Marginal(PointMass(3.0), false, false)

    # One-member blocks become the members' marginals.
    q = rule_marginals(getdata, Val(((:out, :μ), :v)), (split, v))
    @test Set(keys(q)) == Set((:out, :μ, :v))
    @test q[:out] === PointMass(1.0) && q[:μ] === Normal(0.0, 2.0) && q[:v] === PointMass(3.0)

    # A larger block stays a joint.
    joint = MvNormalMeanCovariance([0.0, 1.0], [1.0 0.0; 0.0 1.0])
    q = rule_marginals(getdata, Val(((:out, :μ, :v),)), (Marginal(FactorizedCluster((:out, :μ) => joint, (:v,) => PointMass(2.0)), false, false),))
    @test q[:out, :μ] === joint && q[:v] === PointMass(2.0)

    # Each block carries the joint's annotations.
    ann = rule_marginals(getannotations, Val(((:out, :μ), :v)), (split, v))
    @test ann[:out] isa AnnotationDict && ann[:out] === ann[:μ]

    # A joint that does not split is passed as it is.
    whole = Marginal(joint, false, false)
    @test rule_marginals(getdata, Val(((:out, :μ),)), (whole,))[:out, :μ] === joint
end

@testitem "engine:factorized-cluster:graph" tags = [:engine] setup = [EngineHarness] begin
    # `x := copy(1.0)` sends a point mass into `y ~ NMV(x, v)` under `q(out, μ)q(v)`, so the
    # joint over `out` and `μ` splits (NMV's point-mass marginal rule). The node's free energy
    # reads the blocks: the average energy over the singles, minus the entropy of each block.
    using ExponentialFamily, BayesBase, StandardMessagePassingRules, MessagePassingRulesBase
    import ReactiveMP:
        activate!, getdata, get_node_local_marginals, getlocalclusters, get_stream_of_marginals, score, FactorBoundFreeEnergy,
        FactorNodeActivationOptions, RandomVariableActivationOptions, MessageProductContext, CountingReal
    using Rocket
    H = EngineHarness

    graph = H.Graph()
    x = H.random!(graph)
    μ = H.random!(graph)
    one = H.constant!(graph, 1.0)
    H.node!(graph, H.Copy, [(:out, x), (:in, one)])
    H.node!(graph, NormalMeanVariance, [(:out, μ), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 4.0))])
    likelihood = H.node!(graph, NormalMeanVariance, [(:out, x), (:μ, μ), (:v, H.constant!(graph, 2.0))]; factorisation = ((:out, :μ), (:v,)))

    product = MessageProductContext()
    foreach(v -> activate!(v, RandomVariableActivationOptions(nothing, product, product)), (x, μ))
    foreach(node -> activate!(node, FactorNodeActivationOptions()), graph.nodes)

    joints, energies = [], []
    cluster = first(get_node_local_marginals(getlocalclusters(likelihood)))
    subscriptions = [
        subscribe!(get_stream_of_marginals(cluster), (q) -> push!(joints, getdata(q))),
        subscribe!(score(CountingReal{Float64}, FactorBoundFreeEnergy(), likelihood, nothing, nothing), (f) -> push!(energies, f)),
    ]

    fc = only(joints)
    @test fc isa FactorizedCluster
    # μ's block: its prior N(0, 4) times the likelihood's N(1, 2) given out = 1.
    @test fc[(:out,)] == PointMass(1.0)
    @test mean(fc[(:μ,)]) ≈ (0 / 4 + 1 / 2) / (1 / 4 + 1 / 2) && var(fc[(:μ,)]) ≈ 1 / (1 / 4 + 1 / 2)

    expected = call_average_energy(NormalMeanVariance; q = (out = fc[(:out,)], μ = fc[(:μ,)], v = PointMass(2.0))) - entropy(fc) - entropy(PointMass(2.0))
    @test float(only(energies)) ≈ float(expected)
    foreach(unsubscribe!, subscriptions)
end
