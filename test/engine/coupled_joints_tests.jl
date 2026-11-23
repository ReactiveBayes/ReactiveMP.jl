# Joint clusters of one node that read each other. Each is computed by the node's marginal rule
# from the messages into its members and the other joints' marginals, and nothing outside the node
# writes to any of them.

@testmodule CoupledJoints begin
    using ReactiveMP, Rocket, BayesBase, MessagePassingRulesBase, ExponentialFamily, StandardMessagePassingRules
    import ReactiveMP:
        activate!, getdata, getlocalclusters, get_node_local_marginals, get_stream_of_marginals, set_initial_marginal!,
        FactorNodeActivationOptions, RandomVariableActivationOptions, DataVariableActivationOptions, MessageProductContext

    # A joint is the product of each member's inbound message with N(⋅; μ, 1), for the `means` μ its
    # factor gives the two members. The members are independent under the node's factors.
    function joint(first, second, means)
        members = map((first, second), means) do message, m
            ξ, w = weightedmean_precision(message)
            ((ξ + m) / (w + 1), 1 / (w + 1))
        end
        return MvNormalMeanCovariance([members[1][1], members[2][1]], [members[1][2] 0.0; 0.0 members[2][2]])
    end

    # p(a, b | c, d) = N(a; c, 1) N(b; d, 1), under q(a, b) q(c, d).
    struct TwoPairs end
    @define_factor_node(node = TwoPairs, type = Stochastic, interfaces = [:a, :b, :c, :d])

    # exp(E[log N(a; c, 1)]) ∝ N(a; E[c], 1), and likewise for the other three.
    @define_message_update_rule(node = TwoPairs, target = :a, args = (m[:b]::Any, q[:c, :d]::Any), body = (args) -> NormalMeanVariance(mean(args.q[:c, :d])[1], 1.0))
    @define_message_update_rule(node = TwoPairs, target = :b, args = (m[:a]::Any, q[:c, :d]::Any), body = (args) -> NormalMeanVariance(mean(args.q[:c, :d])[2], 1.0))
    @define_message_update_rule(node = TwoPairs, target = :c, args = (m[:d]::Any, q[:a, :b]::Any), body = (args) -> NormalMeanVariance(mean(args.q[:a, :b])[1], 1.0))
    @define_message_update_rule(node = TwoPairs, target = :d, args = (m[:c]::Any, q[:a, :b]::Any), body = (args) -> NormalMeanVariance(mean(args.q[:a, :b])[2], 1.0))
    @define_marginal_update_rule(
        node = TwoPairs, target = (:a, :b), args = (m[:a]::Any, m[:b]::Any, q[:c, :d]::Any),
        body = (args) -> joint(args.m[:a], args.m[:b], mean(args.q[:c, :d])),
    )
    @define_marginal_update_rule(
        node = TwoPairs, target = (:c, :d), args = (m[:c]::Any, m[:d]::Any, q[:a, :b]::Any),
        body = (args) -> joint(args.m[:c], args.m[:d], mean(args.q[:a, :b])),
    )

    # p(a, b | c, d, e, f) = N(a; c + e, 1) N(b; d + f, 1), under q(a, b) q(c, d) q(e, f).
    struct ThreePairs end
    @define_factor_node(node = ThreePairs, type = Stochastic, interfaces = [:a, :b, :c, :d, :e, :f])

    # exp(E[log N(a; c + e, 1)]) ∝ N(a; E[c] + E[e], 1) towards a, N(c; E[a] - E[e], 1) towards c,
    # N(e; E[a] - E[c], 1) towards e, and likewise for b, d and f.
    towards_ab(qcd, qef) = mean(qcd) + mean(qef)
    towards_cd(qab, qef) = mean(qab) - mean(qef)
    towards_ef(qab, qcd) = mean(qab) - mean(qcd)
    @define_message_update_rule(node = ThreePairs, target = :a, args = (m[:b]::Any, q[:c, :d]::Any, q[:e, :f]::Any), body = (args) -> NormalMeanVariance(towards_ab(args.q[:c, :d], args.q[:e, :f])[1], 1.0))
    @define_message_update_rule(node = ThreePairs, target = :b, args = (m[:a]::Any, q[:c, :d]::Any, q[:e, :f]::Any), body = (args) -> NormalMeanVariance(towards_ab(args.q[:c, :d], args.q[:e, :f])[2], 1.0))
    @define_message_update_rule(node = ThreePairs, target = :c, args = (m[:d]::Any, q[:a, :b]::Any, q[:e, :f]::Any), body = (args) -> NormalMeanVariance(towards_cd(args.q[:a, :b], args.q[:e, :f])[1], 1.0))
    @define_message_update_rule(node = ThreePairs, target = :d, args = (m[:c]::Any, q[:a, :b]::Any, q[:e, :f]::Any), body = (args) -> NormalMeanVariance(towards_cd(args.q[:a, :b], args.q[:e, :f])[2], 1.0))
    @define_message_update_rule(node = ThreePairs, target = :e, args = (m[:f]::Any, q[:a, :b]::Any, q[:c, :d]::Any), body = (args) -> NormalMeanVariance(towards_ef(args.q[:a, :b], args.q[:c, :d])[1], 1.0))
    @define_message_update_rule(node = ThreePairs, target = :f, args = (m[:e]::Any, q[:a, :b]::Any, q[:c, :d]::Any), body = (args) -> NormalMeanVariance(towards_ef(args.q[:a, :b], args.q[:c, :d])[2], 1.0))
    @define_marginal_update_rule(
        node = ThreePairs, target = (:a, :b), args = (m[:a]::Any, m[:b]::Any, q[:c, :d]::Any, q[:e, :f]::Any),
        body = (args) -> joint(args.m[:a], args.m[:b], towards_ab(args.q[:c, :d], args.q[:e, :f])),
    )
    @define_marginal_update_rule(
        node = ThreePairs, target = (:c, :d), args = (m[:c]::Any, m[:d]::Any, q[:a, :b]::Any, q[:e, :f]::Any),
        body = (args) -> joint(args.m[:c], args.m[:d], towards_cd(args.q[:a, :b], args.q[:e, :f])),
    )
    @define_marginal_update_rule(
        node = ThreePairs, target = (:e, :f), args = (m[:e]::Any, m[:f]::Any, q[:a, :b]::Any, q[:c, :d]::Any),
        body = (args) -> joint(args.m[:e], args.m[:f], towards_ef(args.q[:a, :b], args.q[:c, :d])),
    )

    """
    `(a, b) ~ fform(parents...)` with every parent ~ N(0, 1), `oa ~ N(a, 1)` and `ob ~ N(b, 1)`
    observed, activated as RxInfer does and fed `data` once per iteration: `fform` is `TwoPairs`
    (parents c, d) or `ThreePairs` (parents c, d, e, f), under one joint per pair. Every variable
    has two edges, since a variable with one never sends its node a message. `initial_joints`
    seeds every joint of the node. Returns every marginal each variable was given, by name, and
    the number of times each joint was computed.
    """
    function run(fform; data, iterations, initial_joints)
        names = fform === TwoPairs ? (:a, :b, :c, :d) : (:a, :b, :c, :d, :e, :f)
        vars = map(_ -> randomvar(), names)
        oa, ob = datavar(), datavar()
        node(fform, interfaces, factorisation) = factornode(fform, interfaces, factorisation)
        prior(v) = node(NormalMeanVariance, [(:out, v), (:μ, constvar(0.0)), (:v, constvar(1.0))], ((:out,), (:μ,), (:v,)))
        coupled = node(fform, collect(zip(names, vars)), Tuple((names[k], names[k + 1]) for k in 1:2:length(names)))
        nodes = [
            map(prior, vars[3:end])...,
            coupled,
            node(NormalMeanVariance, [(:out, oa), (:μ, vars[1]), (:v, constvar(1.0))], ((:out,), (:μ,), (:v,))),
            node(NormalMeanVariance, [(:out, ob), (:μ, vars[2]), (:v, constvar(1.0))], ((:out,), (:μ,), (:v,))),
        ]
        product = MessageProductContext()
        foreach(v -> activate!(v, RandomVariableActivationOptions(nothing, product, product)), vars)
        foreach(v -> activate!(v, DataVariableActivationOptions(true, false, nothing, nothing)), (oa, ob))
        foreach(n -> activate!(n, FactorNodeActivationOptions()), nodes)

        joints = map(get_stream_of_marginals, get_node_local_marginals(getlocalclusters(coupled)))
        if initial_joints
            foreach(q -> set_initial_marginal!(q, MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0])), joints)
        end

        q = NamedTuple{names}(map(_ -> Any[], names))
        computed = zeros(Int, length(joints))
        subscriptions = Any[]
        for (name, v) in zip(names, vars)
            push!(subscriptions, subscribe!(get_stream_of_marginals(v), (m) -> push!(q[name], getdata(m))))
        end
        for (k, stream) in enumerate(joints)
            push!(subscriptions, subscribe!(stream, (m) -> ReactiveMP.is_initial(m) || (computed[k] += 1)))
        end
        for _ in 1:iterations
            new_observation!(oa, data[1])
            new_observation!(ob, data[2])
        end
        foreach(unsubscribe!, subscriptions)
        return (; q, computed)
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
    result = CoupledJoints.run(CoupledJoints.TwoPairs; data = (3.0, 6.0), iterations = iterations, initial_joints = true)

    @test result.computed == [iterations, iterations]
    @test length(result.q.a) >= iterations
    @test mean(last(result.q.a)) ≈ 2 atol = 1.0e-8
    @test var(last(result.q.a)) ≈ 0.5 atol = 1.0e-8
    @test mean(last(result.q.c)) ≈ 1 atol = 1.0e-8
    @test var(last(result.q.c)) ≈ 0.5 atol = 1.0e-8
end

@testitem "engine:three joints of one node that read each other: coordinate ascent" tags = [:engine] setup = [CoupledJoints] begin
    # Three joints, each reading the other two: q(a, b) is activated before both others, and
    # q(c, d) before q(e, f), so each but the last once waited on streams nothing wrote to. Every
    # marginal has precision 2; with oa = 4, E[a] = (4 + E[c] + E[e])/2, E[c] = (E[a] - E[e])/2 and
    # E[e] = (E[a] - E[c])/2, so the fixed point is E[a] = 3, E[c] = E[e] = 1, and with ob = 8,
    # E[b] = 6, E[d] = E[f] = 2.
    using BayesBase
    iterations = 50
    result = CoupledJoints.run(CoupledJoints.ThreePairs; data = (4.0, 8.0), iterations = iterations, initial_joints = true)

    @test result.computed == [iterations, iterations, iterations]
    for (name, expected) in pairs((a = 3, b = 6, c = 1, d = 2, e = 1, f = 2))
        @test length(result.q[name]) >= iterations
        @test mean(last(result.q[name])) ≈ expected atol = 1.0e-8
        @test var(last(result.q[name])) ≈ 0.5 atol = 1.0e-8
    end
end
