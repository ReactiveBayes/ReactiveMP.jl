# Two joint clusters of one node that read each other. Each is computed by the node's marginal
# rule from the messages into its members and the other joint's marginal, and nothing outside the
# node writes to either.

@testmodule CoupledJoints begin
    using ReactiveMP, Rocket, BayesBase, MessagePassingRulesBase, ExponentialFamily, StandardMessagePassingRules
    import ReactiveMP:
        activate!, israndom, isdata, getdata, getlocalclusters, get_node_local_marginals, get_stream_of_marginals, set_initial_marginal!,
        FactorNodeActivationOptions, RandomVariableActivationOptions, DataVariableActivationOptions, MessageProductContext

    # p(a, b | c, d) = N(a; c, 1) N(b; d, 1), under q(a, b) q(c, d).
    struct TwoPairs end
    @define_factor_node(node = TwoPairs, type = Stochastic, interfaces = [:a, :b, :c, :d])

    # exp(E[log N(a; c, 1)]) ∝ N(a; E[c], 1), and likewise for the other three.
    @define_message_update_rule(node = TwoPairs, target = :a, args = (m[:b]::Any, q[:c, :d]::Any), body = (args) -> NormalMeanVariance(mean(args.q[:c, :d])[1], 1.0))
    @define_message_update_rule(node = TwoPairs, target = :b, args = (m[:a]::Any, q[:c, :d]::Any), body = (args) -> NormalMeanVariance(mean(args.q[:c, :d])[2], 1.0))
    @define_message_update_rule(node = TwoPairs, target = :c, args = (m[:d]::Any, q[:a, :b]::Any), body = (args) -> NormalMeanVariance(mean(args.q[:a, :b])[1], 1.0))
    @define_message_update_rule(node = TwoPairs, target = :d, args = (m[:c]::Any, q[:a, :b]::Any), body = (args) -> NormalMeanVariance(mean(args.q[:a, :b])[2], 1.0))

    # A joint is the product of each member's inbound message with N(⋅; E[other], 1).
    function joint(first, second, other)
        members = map((first, second), mean(other)) do message, m
            ξ, w = weightedmean_precision(message)
            ((ξ + m) / (w + 1), 1 / (w + 1))
        end
        return MvNormalMeanCovariance([members[1][1], members[2][1]], [members[1][2] 0.0; 0.0 members[2][2]])
    end
    @define_marginal_update_rule(
        node = TwoPairs, target = (:a, :b), args = (m[:a]::Any, m[:b]::Any, q[:c, :d]::Any),
        body = (args) -> joint(args.m[:a], args.m[:b], args.q[:c, :d]),
    )
    @define_marginal_update_rule(
        node = TwoPairs, target = (:c, :d), args = (m[:c]::Any, m[:d]::Any, q[:a, :b]::Any),
        body = (args) -> joint(args.m[:c], args.m[:d], args.q[:a, :b]),
    )

    """
    c, d ~ N(0, 1), (a, b) ~ TwoPairs(c, d), oa ~ N(a, 1) and ob ~ N(b, 1) observed, activated as
    RxInfer does and fed `data` once per iteration. `initial_joints` seeds both joints of the node.
    Returns every marginal `a` and `c` were given and the number of times each joint was computed.
    """
    function run(; data, iterations, initial_joints)
        a, b, c, d = (randomvar() for _ in 1:4)
        oa, ob = datavar(), datavar()
        node(fform, interfaces, factorisation) = factornode(fform, interfaces, factorisation)
        nodes = [
            node(NormalMeanVariance, [(:out, c), (:μ, constvar(0.0)), (:v, constvar(1.0))], ((:out,), (:μ,), (:v,))),
            node(NormalMeanVariance, [(:out, d), (:μ, constvar(0.0)), (:v, constvar(1.0))], ((:out,), (:μ,), (:v,))),
            node(TwoPairs, [(:a, a), (:b, b), (:c, c), (:d, d)], ((:a, :b), (:c, :d))),
            node(NormalMeanVariance, [(:out, oa), (:μ, a), (:v, constvar(1.0))], ((:out,), (:μ,), (:v,))),
            node(NormalMeanVariance, [(:out, ob), (:μ, b), (:v, constvar(1.0))], ((:out,), (:μ,), (:v,))),
        ]
        product = MessageProductContext()
        foreach(v -> activate!(v, RandomVariableActivationOptions(nothing, product, product)), (a, b, c, d))
        foreach(v -> activate!(v, DataVariableActivationOptions(true, false, nothing, nothing)), (oa, ob))
        foreach(n -> activate!(n, FactorNodeActivationOptions()), nodes)

        joints = map(get_stream_of_marginals, get_node_local_marginals(getlocalclusters(nodes[3])))
        if initial_joints
            foreach(q -> set_initial_marginal!(q, MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0])), joints)
        end

        qa, qc, computed = Any[], Any[], [0, 0]
        subscriptions = Any[
            subscribe!(get_stream_of_marginals(a), (q) -> push!(qa, getdata(q))),
            subscribe!(get_stream_of_marginals(c), (q) -> push!(qc, getdata(q))),
            subscribe!(joints[1], (q) -> ReactiveMP.is_initial(q) || (computed[1] += 1)),
            subscribe!(joints[2], (q) -> ReactiveMP.is_initial(q) || (computed[2] += 1)),
        ]
        for _ in 1:iterations
            new_observation!(oa, data[1])
            new_observation!(ob, data[2])
        end
        foreach(unsubscribe!, subscriptions)
        return (; qa, qc, computed)
    end
end

@testitem "engine:two joints of one node that read each other: coordinate ascent" tags = [:engine] setup = [CoupledJoints] begin
    # Each joint reads the other, so both are seeded; from there every iteration is one sweep of
    # coordinate ascent. With oa = 3: q(a) has precision 2 and mean (3 + E[c])/2, q(c) precision 2
    # and mean E[a]/2, so the fixed point is E[a] = 2, E[c] = 1.
    #
    # Activating q(a, b) takes q(c, d)'s stream as its input before q(c, d) is activated. That
    # activation once replaced the stream with a new one, the one seeded and computed, and left
    # q(a, b) waiting on a stream nothing wrote to: the messages fired once on the seeds and
    # inference stopped.
    using BayesBase
    iterations = 30
    result = CoupledJoints.run(data = (3.0, 6.0), iterations = iterations, initial_joints = true)

    @test result.computed == [iterations, iterations]
    @test length(result.qa) >= iterations
    @test mean(last(result.qa)) ≈ 2 atol = 1.0e-8
    @test var(last(result.qa)) ≈ 0.5 atol = 1.0e-8
    @test mean(last(result.qc)) ≈ 1 atol = 1.0e-8
    @test var(last(result.qc)) ≈ 0.5 atol = 1.0e-8
end
