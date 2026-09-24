@testmodule DependencySchemeNodes begin
    using MessagePassingRulesBase, ExponentialFamily, Distributions

    struct ThreeInterfaces end
    @define_factor_node(node = ThreeInterfaces, type = Stochastic, interfaces = [:a, :b, :c])

    struct DeclaresDependencies end
    @define_factor_node(
        node = DeclaresDependencies, type = Stochastic, interfaces = [:out, :in],
        dependencies = [:out => (m[:in],), :in => (m[:out],)],
    )

    struct WithGroup end
    @define_factor_node(node = WithGroup, type = Stochastic, interfaces = [:out, :m...])

    # Every selector, under the node's own algorithm.
    struct Selectors end
    struct SelectorsAlgorithm <: AbstractAlgorithm end
    @define_factor_node(
        node = Selectors, type = Stochastic, interfaces = [:out, :m..., :p...], algorithm = SelectorsAlgorithm,
        dependencies = [
            :out => (q[:p...], m[:m...]),
            (:m, k) => (q[:out], q[:p][k], m[:m][!k]),
            (:p, k) => (q[:m][k],),
        ],
    )

    struct FixedPartition end
    struct FixedPartitionAlgorithm <: AbstractAlgorithm end
    @define_factor_node(node = FixedPartition, type = Stochastic, interfaces = [:out, :μ, :τ], algorithm = FixedPartitionAlgorithm)
    @define_dependencies(
        node = FixedPartition, algorithm = FixedPartitionAlgorithm,
        dependencies = [:out => (q[:μ], q[:τ]), :μ => (q[:out], q[:τ]), :τ => (q[:out], q[:μ])],
        free_energy_partition = [(:out,), (:μ,), (:τ,)],
    )

    # `y ~ N(a x, 1/W)` with the rule towards `a` reading `q(a)` beside the default scheme's
    # inputs, as ContinuousTransition's does; `Redundant` adds inputs the default already has,
    # or a message.
    struct Transition end
    struct TransitionVMP <: AbstractAlgorithm end
    struct Redundant <: AbstractAlgorithm end
    @define_factor_node(
        node = Transition, type = Stochastic, interfaces = [:y, :x, :a, :W], algorithm = TransitionVMP,
        dependencies = [:y => (default,), :x => (default,), :a => (default, q[:a]), :W => (default,)],
    )
    # Its mean-field rules, the one towards `a` counting its calls.
    const READS_OF_A = Ref(0)
    @define_message_update_rule(
        node = Transition, target = :x, args = (q[:y]::Any, q[:a]::Any, q[:W]::Any),
        body = (args) -> NormalWeightedMeanPrecision(mean(args.q[:a]) * mean(args.q[:W]) * mean(args.q[:y]), (mean(args.q[:a])^2 + var(args.q[:a])) * mean(args.q[:W])),
    )
    @define_message_update_rule(
        node = Transition, target = :a, args = (q[:y]::Any, q[:x]::Any, q[:a]::Any, q[:W]::Any),
        body = (args) -> begin
            READS_OF_A[] += 1
            NormalWeightedMeanPrecision(mean(args.q[:x]) * mean(args.q[:W]) * mean(args.q[:y]), (mean(args.q[:x])^2 + var(args.q[:x])) * mean(args.q[:W]))
        end,
    )
    @define_message_update_rule(
        node = Transition, target = :W, args = (q[:y]::Any, q[:x]::Any, q[:a]::Any),
        body = (args) -> GammaShapeRate(1.5, (mean(args.q[:y]) - mean(args.q[:a]) * mean(args.q[:x]))^2 / 2 + 0.5),
    )
    @define_dependencies(
        node = Transition, algorithm = Redundant,
        dependencies = [:y => (default, q[:a]), :x => (default, m[:x]), :a => (default, q[:W], q[:a]), :W => (default, m[:y])],
    )
end

@testitem "collect_latest_messages" tags = [:nodes] setup = [DependencySchemeNodes] begin
    include("../testutilities.jl")
    using BayesBase

    import ReactiveMP: collect_latest_messages, getdata, getrecent

    node = factornode(
        DependencySchemeNodes.ThreeInterfaces,
        [(:a, ConstVariable(1)), (:b, ConstVariable(2)), (:c, ConstVariable(3))],
    )
    a, b, c = getinterfaces(node)

    @testset let (tag, stream) = collect_latest_messages(())
        @test tag === nothing
        @test check_stream_updated_once(stream) === nothing
    end

    for (interfaces, names, values) in (
            ((a, b, c), (:a, :b, :c), (PointMass(1), PointMass(2), PointMass(3))),
            ((a, b), (:a, :b), (PointMass(1), PointMass(2))),
            ((b, c), (:b, :c), (PointMass(2), PointMass(3))),
            ((a, c), (:a, :c), (PointMass(1), PointMass(3))),
        )
        @testset let (tag, stream) = collect_latest_messages(interfaces)
            @test tag === Val(names)
            @test getdata.(getrecent.(check_stream_updated_once(stream))) === values
        end
    end
end

@testitem "collect_latest_marginals" tags = [:nodes] setup = [DependencySchemeNodes] begin
    include("../testutilities.jl")
    using BayesBase

    import ReactiveMP:
        collect_latest_marginals, getdata, getrecent, getlocalclusters, get_stream_of_marginals,
        set_stream_of_marginals!, get_node_local_marginals

    a_v, b_v, c_v = ConstVariable(1), ConstVariable(2), ConstVariable(3)
    node = factornode(
        DependencySchemeNodes.ThreeInterfaces, [(:a, a_v), (:b, b_v), (:c, c_v)], ((:a,), (:b,), (:c,)),
    )
    a, b, c = get_node_local_marginals(getlocalclusters(node))

    set_stream_of_marginals!(a, get_stream_of_marginals(a_v))
    set_stream_of_marginals!(b, get_stream_of_marginals(b_v))
    set_stream_of_marginals!(c, get_stream_of_marginals(c_v))

    for (marginals, names, values) in (
            ((a, b, c), (:a, :b, :c), (PointMass(1), PointMass(2), PointMass(3))),
            ((a, b), (:a, :b), (PointMass(1), PointMass(2))),
            ((b, c), (:b, :c), (PointMass(2), PointMass(3))),
            ((a, c), (:a, :c), (PointMass(1), PointMass(3))),
        )
        @testset let (tag, stream) = collect_latest_marginals(marginals)
            @test tag === Val(names)
            @test getdata.(getrecent.(check_stream_updated_once(stream))) === values
        end
    end
end

@testitem "collect_latest_marginals should re-fire while all dependencies are initial (deadlock guard, RxInfer#344)" tags = [:nodes] setup = [DependencySchemeNodes] begin
    # Regression test for https://github.com/ReactiveBayes/RxInfer.jl/issues/344
    # With plain `PushNew()` semantics every marginal dependency must refresh before the
    # combined stream may fire again. If the first firing consumed only provisional
    # (`is_initial`) marginals, structured VMP with 3+ mutually-dependent clusters deadlocks:
    # the combination never fires again because the clusters wait on each other.
    # `reset_vstatus_of_sources` must keep the combination hot while all of its recent
    # values are `is_initial`, and strict `PushNew()` semantics must be restored as soon as
    # at least one dependency holds a real (non-initial) value.
    include("../testutilities.jl")
    using BayesBase, Rocket

    import ReactiveMP: collect_latest_marginals, getlocalclusters, set_stream_of_marginals!, get_node_local_marginals

    node = factornode(
        DependencySchemeNodes.ThreeInterfaces,
        [(:a, randomvar()), (:b, randomvar()), (:c, randomvar())],
        ((:a,), (:b,), (:c,)),
    )
    a, b, c = get_node_local_marginals(getlocalclusters(node))

    sa, sb, sc = RecentSubject(Marginal), RecentSubject(Marginal), RecentSubject(Marginal)
    set_stream_of_marginals!(a, sa)
    set_stream_of_marginals!(b, sb)
    set_stream_of_marginals!(c, sc)

    (tag, stream) = collect_latest_marginals((a, b, c))

    updates = Ref(0)
    subscription = subscribe!(stream, (_) -> updates[] += 1)

    initial(value) = Marginal(value, false, true)
    non_initial(value) = Marginal(value, false, false)

    # The combination fires for the first time only once all dependencies emitted
    next!(sa, initial(PointMass(1)))
    next!(sb, initial(PointMass(2)))
    @test updates[] == 0
    next!(sc, initial(PointMass(3)))
    @test updates[] == 1

    # All consumed values were `is_initial`, thus a single refreshed dependency must be
    # enough to re-fire the combination (this is the #344 deadlock fix)
    next!(sa, non_initial(PointMass(1)))
    @test updates[] == 2

    # The last firing consumed a real value for `a`, thus strict `PushNew()` semantics
    # apply again: the combination must not re-fire until all dependencies refresh
    next!(sb, non_initial(PointMass(2)))
    @test updates[] == 2
    next!(sc, non_initial(PointMass(3)))
    @test updates[] == 2
    next!(sa, non_initial(PointMass(1)))
    @test updates[] == 3

    # And a single refresh alone keeps being insufficient
    next!(sb, non_initial(PointMass(2)))
    @test updates[] == 3

    unsubscribe!(subscription)
end

@testitem "collect_latest_messages should keep strict PushNew semantics even for initial messages" tags = [:nodes] setup = [DependencySchemeNodes] begin
    # Guards the scoping of the RxInfer#344 deadlock fix: the `is_initial` vstatus reset is
    # deliberately applied to marginal dependencies only. Applying it to message dependencies
    # as well changes the message-update schedule in models unaffected by the deadlock
    # (an outbound message would be recomputed as soon as a single dependency refreshes while
    # the others are still `is_initial`), which changes free-energy trajectories and breaks
    # strict FE-monotonicity guarantees downstream (observed in RxInfer model tests).
    include("../testutilities.jl")
    using BayesBase, Rocket

    import ReactiveMP: MessageObservable, collect_latest_messages, getvariable, connect!

    node = factornode(
        DependencySchemeNodes.ThreeInterfaces, [(:a, randomvar()), (:b, randomvar()), (:c, randomvar())],
    )
    a, b, c = getinterfaces(node)

    sa, sb, sc = RecentSubject(Message), RecentSubject(Message), RecentSubject(Message)

    # Wire the variables' outbound message streams (the inbound message streams of the
    # interfaces) manually, the same way `activate!` would do
    for (interface, subject) in zip((a, b, c), (sa, sb, sc))
        output = MessageObservable(Message)
        connect!(output, subject)
        push!(getvariable(interface).output_messages, output)
    end

    (tag, stream) = collect_latest_messages((a, b, c))

    updates = Ref(0)
    subscription = subscribe!(stream, (_) -> updates[] += 1)

    initial(value) = Message(value, false, true)
    non_initial(value) = Message(value, false, false)

    # The combination fires for the first time only once all dependencies emitted
    next!(sa, initial(PointMass(1)))
    next!(sb, initial(PointMass(2)))
    @test updates[] == 0
    next!(sc, initial(PointMass(3)))
    @test updates[] == 1

    # Even though all consumed values were `is_initial`, message dependencies must keep
    # strict `PushNew()` semantics: a single refreshed dependency is not enough to re-fire
    next!(sa, non_initial(PointMass(1)))
    @test updates[] == 1
    next!(sb, non_initial(PointMass(2)))
    @test updates[] == 1
    next!(sc, non_initial(PointMass(3)))
    @test updates[] == 2

    unsubscribe!(subscription)
end

@testitem "default_dependencies follow the factorisation" tags = [:nodes] setup = [DependencySchemeNodes] begin
    import ReactiveMP: default_dependencies, name

    variables() = [(:a, randomvar()), (:b, randomvar()), (:c, randomvar())]
    dependencies(node, i) = map(first, default_dependencies(node, i))

    @testset "one cluster: the other messages of the cluster, no marginals" begin
        node = factornode(DependencySchemeNodes.ThreeInterfaces, variables())
        @test dependencies(node, 1) == ((:b, :c), ())
        @test dependencies(node, 2) == ((:a, :c), ())
        @test dependencies(node, 3) == ((:a, :b), ())
    end

    @testset "mean field: no messages, the marginals of the other clusters" begin
        node = factornode(DependencySchemeNodes.ThreeInterfaces, variables(), ((:a,), (:b,), (:c,)))
        @test dependencies(node, 1) == ((), (:b, :c))
        @test dependencies(node, 2) == ((), (:a, :c))
        @test dependencies(node, 3) == ((), (:a, :b))
    end

    @testset "structured: a joint is read by the tuple of its members" begin
        node = factornode(DependencySchemeNodes.ThreeInterfaces, variables(), ((:c,), (:a, :b)))
        @test dependencies(node, 1) == ((:b,), (:c,))
        @test dependencies(node, 2) == ((:a,), (:c,))
        @test dependencies(node, 3) == ((), ((:a, :b),))
    end
end

@testitem "input_names folds a group's members into one input" tags = [:nodes] begin
    import ReactiveMP: input_names, GroupMember, GroupInputs, rule_messages, rule_marginals

    @test input_names((:out, (:y, :x))) === Val{(:out, (:y, :x))}()
    @test input_names((:out, GroupMember(:p, 1, 3), GroupMember(:p, 3, 3), GroupMember(:m, 2, 3))) ===
        Val{(:out, GroupInputs{:p, 3, (1, 3)}(), GroupInputs{:m, 3, (2,)}())}()
    @test_throws "more than once" input_names((GroupMember(:p, 1, 2), :out, GroupMember(:p, 2, 2)))
    @test_throws "member order" input_names((GroupMember(:p, 2, 2), GroupMember(:p, 1, 2)))

    # A group reaches the rule full length, `nothing` where a member is not an input.
    names = input_names((:out, GroupMember(:p, 2, 3), (:y, :x)))
    q = rule_marginals(identity, names, (1.0, 2.0, 3.0))
    @test q[:out] === 1.0 && q[:p] === (nothing, 2.0, nothing) && q[:y, :x] === 3.0
    @test rule_messages(identity, input_names((GroupMember(:m, 1, 2), GroupMember(:m, 2, 2))), (4.0, 5.0))[:m] === (4.0, 5.0)
end

@testitem "declared dependencies select group members relative to the target" tags = [:nodes] setup = [DependencySchemeNodes] begin
    import ReactiveMP: declared_dependencies, getinterfaces, getvariable, GroupMember, name, index
    import MessagePassingRulesBase: dependencies_spec
    using .DependencySchemeNodes: Selectors, SelectorsAlgorithm

    out = randomvar()
    m, p = [randomvar() for _ in 1:3], [randomvar() for _ in 1:3]
    node = factornode(Selectors, [(:out, out), (((:m, k), m[k]) for k in 1:3)..., (((:p, k), p[k]) for k in 1:3)...])
    spec = dependencies_spec(Selectors, SelectorsAlgorithm())
    interfaces = getinterfaces(node)
    member(group, k) = only(filter(i -> name(i) === group && i isa ReactiveMP.IndexedNodeInterface && index(i) == k, interfaces))

    # `:out => (q[:p...], m[:m...])`: every member, in member order.
    (messagelabels, messages), (marginallabels, marginals) = declared_dependencies(node, spec, first(interfaces))
    @test messagelabels == Tuple(GroupMember(:m, k, 3) for k in 1:3)
    @test messages == Tuple(member(:m, k) for k in 1:3)
    @test marginallabels == Tuple(GroupMember(:p, k, 3) for k in 1:3)
    @test all(marginals .=== Tuple(p))

    # `(:m, 2) => (q[:out], q[:p][k], m[:m][!k])`: the aligned member, and every other one.
    (messagelabels, messages), (marginallabels, marginals) = declared_dependencies(node, spec, member(:m, 2))
    @test messagelabels == (GroupMember(:m, 1, 3), GroupMember(:m, 3, 3))
    @test marginallabels == (:out, GroupMember(:p, 2, 3))
    @test marginals[1] === out && marginals[2] === p[2]
end

@testitem "declared dependencies can extend the default scheme" tags = [:nodes] setup = [DependencySchemeNodes] begin
    import ReactiveMP: declared_dependencies, default_dependencies, getinterfaces, getvariable, name
    import MessagePassingRulesBase: dependencies_spec
    using .DependencySchemeNodes: Transition, TransitionVMP, Redundant

    variables() = [(:y, randomvar()), (:x, randomvar()), (:a, randomvar()), (:W, randomvar())]
    labels(node, spec, i) = map(first, declared_dependencies(node, spec, getinterfaces(node)[i]))
    spec, redundant = dependencies_spec(Transition, TransitionVMP()), dependencies_spec(Transition, Redundant())

    @testset "mean field: q(a) among the other marginals, in interface order" begin
        interfaces = variables()
        node = factornode(Transition, interfaces, ((:y,), (:x,), (:a,), (:W,)))
        (_, messages), (_, marginals) = declared_dependencies(node, spec, getinterfaces(node)[3])
        @test labels(node, spec, 3) == ((), (:y, :x, :a, :W))
        @test isempty(messages)
        # The auxiliary input is the variable's own marginal, not a cluster of the node's.
        @test marginals[3] === last(interfaces[3])
        # `default` alone is the default scheme.
        for i in (1, 2, 4)
            @test labels(node, spec, i) == map(first, default_dependencies(node, i))
        end
    end

    @testset "structured: q(a) after the joint q(y, x)" begin
        node = factornode(Transition, variables(), ((:y, :x), (:a,), (:W,)))
        @test labels(node, spec, 3) == ((), ((:y, :x), :a, :W))
        @test labels(node, spec, 2) == ((:y,), (:a, :W))
        @test labels(node, spec, 4) == ((), ((:y, :x), :a))
    end

    @testset "an input the default scheme has is not added twice; a message goes among the messages" begin
        node = factornode(Transition, variables(), ((:y,), (:x,), (:a,), (:W,)))
        @test labels(node, redundant, 1) == ((), (:x, :a, :W))
        @test labels(node, redundant, 3) == ((), (:y, :x, :a, :W))
        @test labels(node, redundant, 2) == ((:x,), (:y, :a, :W))
        structured = factornode(Transition, variables(), ((:y, :x), (:a,), (:W,)))
        # `x`'s own message after `y`'s, which the joint already gives.
        @test labels(structured, redundant, 2) == ((:y, :x), (:a, :W))
        @test labels(structured, redundant, 4) == ((:y,), ((:y, :x), :a))
    end
end

@testitem "a rule that reads its own target's marginal runs once per iteration" tags = [:nodes] setup = [DependencySchemeNodes, EngineHarness] begin
    # y ~ N(a x, 1/W) observed, under mean-field: the message towards `a` updates q(a), which it
    # reads. `PushNew()` recomputes a message only once all its inputs have refreshed, so the
    # cycle m(→a) → q(a) → m(→a) does not recurse, as in v6.
    using ExponentialFamily, BayesBase, StandardMessagePassingRules
    import ReactiveMP: getlocalclusters, get_node_local_marginals
    using .DependencySchemeNodes: Transition
    H = EngineHarness

    graph = H.Graph()
    y, x, a, W = H.data!(graph), H.random!(graph), H.random!(graph), H.random!(graph)
    H.node!(graph, NormalMeanVariance, [(:out, x), (:μ, H.constant!(graph, 1.0)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, NormalMeanVariance, [(:out, a), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, GammaShapeRate, [(:out, W), (:α, H.constant!(graph, 2.0)), (:β, H.constant!(graph, 1.0))])
    transition = H.node!(graph, Transition, [(:y, y), (:x, x), (:a, a), (:W, W)]; factorisation = ((:y,), (:x,), (:a,), (:W,)))

    trajectory = H.run(
        graph; id = "own marginal", data = [y => 2.0], iterations = 4, posteriors = [:a => a, :x => x, :W => W], free_energy = false,
        initial_marginals = [x => NormalMeanVariance(1.0, 1.0), a => NormalMeanVariance(0.5, 1.0), W => GammaShapeRate(2.0, 1.0)],
    )
    # One message towards each of x, a and W per iteration, and each prior's once.
    calls(target) = [count(r -> r.node == "Transition" && r.target == target && r.iteration == it, trajectory.trace) for it in 1:4]
    @test calls(":a") == [1, 1, 1, 1]
    @test calls(":x") == [1, 1, 1, 1]
    @test calls(":W") == [1, 1, 1, 1]
    @test DependencySchemeNodes.READS_OF_A[] == 4
    # q(a) is consumed, never a cluster of the node, so free energy does not score it twice.
    @test length(get_node_local_marginals(getlocalclusters(transition))) == 4
end

@testitem "activate! wires declared dependencies, groups and a declared partition" tags = [:nodes] setup = [DependencySchemeNodes] begin
    import ReactiveMP: activate!, FactorNodeActivationOptions, RandomVariableActivationOptions, MessageProductContext
    using .DependencySchemeNodes: Selectors, SelectorsAlgorithm, FixedPartition, FixedPartitionAlgorithm, WithGroup

    # As RxInfer does: the node is created, then its variables are activated, then the node.
    function activated(node, interfaces)
        foreach(((_, variable),) -> activate!(variable, RandomVariableActivationOptions(nothing, MessageProductContext(), MessageProductContext())), interfaces)
        return node
    end
    selectors = [(:out, randomvar()), (((:m, k), randomvar()) for k in 1:2)..., (((:p, k), randomvar()) for k in 1:2)...]
    node = activated(factornode(Selectors, selectors, Tuple((key,) for (key, _) in selectors)), selectors)
    @test activate!(node, FactorNodeActivationOptions(; algorithm = SelectorsAlgorithm())) === nothing

    # A group under the default scheme is wired too, its members folded into one input.
    grouped = factornode(WithGroup, [(:out, randomvar()), ((:m, 1), randomvar()), ((:m, 2), randomvar())], ((:out,), ((:m, 1),), ((:m, 2),)))
    @test activate!(grouped, FactorNodeActivationOptions()) === nothing

    # A declared partition must be the factorisation.
    partitioned() = [(:out, randomvar()), (:μ, randomvar()), (:τ, randomvar())]
    @test activate!(factornode(FixedPartition, partitioned(), ((:out,), (:μ,), (:τ,))), FactorNodeActivationOptions()) === nothing
    @test_throws "declares the free-energy partition" activate!(factornode(FixedPartition, partitioned(), ((:out, :μ), (:τ,))), FactorNodeActivationOptions())

    # A joint may hold a whole group, keyed by its name, but not only some of its members.
    whole = factornode(WithGroup, [(:out, randomvar()), ((:m, 1), randomvar()), ((:m, 2), randomvar())], ((:out,), ((:m, 1), (:m, 2))))
    @test map(ReactiveMP.name, ReactiveMP.get_node_local_marginals(ReactiveMP.getlocalclusters(whole))) == (:out, (:m,))
    joint = factornode(WithGroup, [(:out, randomvar()), ((:m, 1), randomvar()), ((:m, 2), randomvar())], ((:out, (:m, 1)), ((:m, 2),)))
    @test_throws "joins some members of a group" activate!(joint, FactorNodeActivationOptions())
end
