@testmodule EngineNodes begin
    using MessagePassingRulesBase

    struct Gaussian end
    @define_factor_node(node = Gaussian, type = Stochastic, interfaces = [:out, (:μ, aliases = [:mean]), (:v, aliases = [:var])])

    function shift end
    @define_factor_node(node = shift, type = Deterministic, interfaces = [:out, :in])

    struct Mixture end
    @define_factor_node(node = Mixture, type = Stochastic, interfaces = [:out, :switch, :m...])

    struct NotANode end
end

@testitem "factornode keeps the node type and its interfaces" tags = [:nodes] setup = [EngineNodes] begin
    import ReactiveMP: functionalform, getinterfaces, getinterface, name
    N = EngineNodes

    gaussian = factornode(N.Gaussian, [(:out, randomvar()), (:μ, randomvar()), (:v, constvar(1.0))])
    @test @inferred(functionalform(gaussian)) === N.Gaussian
    @test name.(getinterfaces(gaussian)) == [:out, :μ, :v]
    @test name(getinterface(gaussian, 2)) === :μ

    shifted = factornode(N.shift, [(:out, randomvar()), (:in, randomvar())])
    @test @inferred(functionalform(shifted)) === N.shift
    @test name.(getinterfaces(shifted)) == [:out, :in]
end

@testitem "sdtype comes from the node declaration" tags = [:nodes] setup = [EngineNodes] begin
    N = EngineNodes

    @test isdeterministic(Deterministic()) && isdeterministic(Deterministic)
    @test !isdeterministic(Stochastic()) && !isdeterministic(Stochastic)
    @test isstochastic(Stochastic()) && isstochastic(Stochastic)
    @test !isstochastic(Deterministic()) && !isstochastic(Deterministic)

    @test sdtype(N.Gaussian) === Stochastic()
    @test sdtype(N.shift) === Deterministic()
    @test sdtype(factornode(N.shift, [(:out, randomvar()), (:in, randomvar())])) === Deterministic()
    # The engine's and the base package's are the same types
    @test Stochastic === ReactiveMP.MessagePassingRulesBase.Stochastic
end

@testitem "factornode puts the interfaces in declaration order and resolves aliases" tags = [:nodes] setup = [EngineNodes] begin
    import ReactiveMP: getinterfaces, getvariable, name
    N = EngineNodes

    out, μ, v = randomvar(), randomvar(), randomvar()
    node = factornode(N.Gaussian, [(:var, v), (:out, out), (:mean, μ)])
    @test name.(getinterfaces(node)) == [:out, :μ, :v]
    @test getvariable.(getinterfaces(node)) == [out, μ, v]
end

@testitem "factornode takes a group's members by index" tags = [:nodes] setup = [EngineNodes] begin
    import ReactiveMP: getinterfaces, getvariable, name, index, IndexedNodeInterface
    N = EngineNodes

    out, switch, m1, m2, m3 = randomvar(), randomvar(), randomvar(), randomvar(), randomvar()
    node = factornode(N.Mixture, [((:m, 2), m2), (:out, out), ((:m, 3), m3), (:switch, switch), ((:m, 1), m1)])
    interfaces = getinterfaces(node)
    @test name.(interfaces) == [:out, :switch, :m, :m, :m]
    @test all(i -> i isa IndexedNodeInterface, interfaces[3:5])
    @test index.(interfaces[3:5]) == [1, 2, 3]
    @test getvariable.(interfaces) == [out, switch, m1, m2, m3]

    @test_throws "must be `1:n`" factornode(N.Mixture, [(:out, out), (:switch, switch), ((:m, 2), m2)])
    @test_throws "needs at least one member" factornode(N.Mixture, [(:out, out), (:switch, switch)])
end

@testitem "factornode checks the interfaces it is given" tags = [:nodes] setup = [EngineNodes] begin
    N = EngineNodes
    out, μ, v, w = randomvar(), randomvar(), randomvar(), randomvar()

    @test_throws "is not a factor node" factornode(N.NotANode, [(:out, out)])
    @test_throws "at least one interface" factornode(N.Gaussian, [])
    @test_throws "needs a variable for its interface `v`" factornode(N.Gaussian, [(:out, out), (:μ, μ)])
    @test_throws "has no interface or alias `w`" factornode(N.Gaussian, [(:out, out), (:μ, μ), (:v, v), (:w, w)])
    @test_throws r"duplicate entry for interface `:μ`. Did you pass an array \(e.g. `x`\) instead of an array element \(e\.g\. `x\[i\]`\)\?" factornode(
        N.Gaussian, [(:out, out), (:μ, μ), (:mean, w), (:v, v)],
    )
    @test_throws "is `:name` or `(:group, k)`" factornode(N.Gaussian, [(1, out), (:μ, μ), (:v, v)])
end

@testitem "factornode reads the factorisation by interface names" tags = [:nodes] setup = [EngineNodes] begin
    import ReactiveMP: getlocalclusters, getfactorization, get_node_local_marginals, name
    N = EngineNodes
    variables() = [(:out, randomvar()), (:μ, randomvar()), (:v, randomvar())]
    clusters(node) = getfactorization(getlocalclusters(node))
    keys(node) = map(name, get_node_local_marginals(getlocalclusters(node)))

    full = factornode(N.Gaussian, variables())
    @test clusters(full) == ((1, 2, 3),)
    @test keys(full) == ((:out, :μ, :v),)

    meanfield = factornode(N.Gaussian, variables(), ((:out,), (:μ,), (:v,)))
    @test clusters(meanfield) == ((1,), (2,), (3,))
    @test keys(meanfield) == (:out, :μ, :v)

    # Members are sorted into declaration order, clusters by their first member; aliases work
    structured = factornode(N.Gaussian, variables(), ((:var,), (:mean, :out)))
    @test clusters(structured) == ((1, 2), (3,))
    @test keys(structured) == ((:out, :μ), :v)

    # A deterministic node's clusters are its output and the joint over its inputs, whatever
    # the factorisation says
    shifted = factornode(N.shift, [(:out, randomvar()), (:in, randomvar())], ((:out, :in),))
    @test clusters(shifted) == ((1,), (2,))

    grouped = factornode(N.Mixture, [(:out, randomvar()), (:switch, randomvar()), ((:m, 1), randomvar()), ((:m, 2), randomvar())], ((:out,), (:switch,), ((:m, 1),), ((:m, 2),)))
    @test clusters(grouped) == ((1,), (2,), (3,), (4,))

    @test_throws "names `(:m, 3)`, which is not one of its interfaces" factornode(
        N.Mixture, [(:out, randomvar()), (:switch, randomvar()), ((:m, 1), randomvar())], ((:out, :switch, (:m, 1)), ((:m, 3),)),
    )
    @test_throws "has no interface or alias `w`" factornode(N.Gaussian, variables(), ((:out, :μ, :v), (:w,)))
    @test_throws "is in more than one cluster" factornode(N.Gaussian, variables(), ((:out, :μ), (:μ, :v)))
    @test_throws "does not cover :v" factornode(N.Gaussian, variables(), ((:out, :μ),))
end
