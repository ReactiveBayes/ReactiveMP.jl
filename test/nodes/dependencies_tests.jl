@testmodule DependencySchemeNodes begin
    using MessagePassingRulesBase

    struct ThreeInterfaces end
    @define_factor_node(node = ThreeInterfaces, type = Stochastic, interfaces = [:a, :b, :c])

    struct DeclaresDependencies end
    @define_factor_node(
        node = DeclaresDependencies, type = Stochastic, interfaces = [:out, :in],
        dependencies = [:out => (m[:in],), :in => (m[:out],)],
    )

    struct WithGroup end
    @define_factor_node(node = WithGroup, type = Stochastic, interfaces = [:out, :m...])
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
    dependencies(node, i) = map(names -> map(name, names), default_dependencies(node, i))

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

@testitem "activate! wires only the default scheme, for now" tags = [:nodes] setup = [DependencySchemeNodes] begin
    import ReactiveMP: activate!, FactorNodeActivationOptions

    declared = factornode(DependencySchemeNodes.DeclaresDependencies, [(:out, randomvar()), (:in, randomvar())])
    @test_throws "declares its own dependencies" activate!(declared, FactorNodeActivationOptions())

    grouped = factornode(DependencySchemeNodes.WithGroup, [(:out, randomvar()), ((:m, 1), randomvar()), ((:m, 2), randomvar())])
    @test_throws "interface group" activate!(grouped, FactorNodeActivationOptions())
end
