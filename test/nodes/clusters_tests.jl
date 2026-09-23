@testitem "FactorNodeLocalMarginal" tags = [:nodes] begin
    import Rocket: of, subscribe!, unsubscribe!
    import ReactiveMP:
        FactorNodeLocalMarginal,
        MarginalObservable,
        get_stream_of_marginals,
        set_stream_of_marginals!,
        name

    @testset let localmarginal = FactorNodeLocalMarginal(:a)
        @test name(localmarginal) === :a
        @test occursin("a", repr(localmarginal))
        # The stream is not set
        @test_throws UndefRefError get_stream_of_marginals(localmarginal)

        m = MarginalObservable()

        set_stream_of_marginals!(localmarginal, m)

        @test get_stream_of_marginals(localmarginal) === m
    end

    @testset let localmarginal = FactorNodeLocalMarginal((:a, :b))
        @test name(localmarginal) === (:a, :b)
        @test repr(localmarginal) == "FactorNodeLocalMarginal((:a, :b))"
        # The stream is not set
        @test_throws UndefRefError get_stream_of_marginals(localmarginal)

        m = of(Marginal("message", false, false))

        set_stream_of_marginals!(localmarginal, m)

        @test get_stream_of_marginals(localmarginal) !== m

        stream_of_marginals = get_stream_of_marginals(localmarginal)

        output_value = []

        subscription = subscribe!(
            stream_of_marginals, (d) -> push!(output_value, d)
        )

        @test length(output_value) === 1
        @test output_value[1] == Marginal("message", false, false)

        unsubscribe!(subscription)
    end
end

@testitem "FactorNodeLocalClusters constructor" tags = [:nodes] begin
    import ReactiveMP:
        NodeInterface,
        FactorNodeLocalClusters,
        getfactorization,
        get_node_local_marginals,
        name

    a = NodeInterface(:a, randomvar())
    b = NodeInterface(:b, randomvar())
    c = NodeInterface(:c, randomvar())

    # Interfaces can be both tuples and arrays
    for interfaces in [(a, b, c), [a, b, c]]
        @testset let clusters = FactorNodeLocalClusters(
                interfaces, ((1, 2, 3),)
            )
            @test length(get_node_local_marginals(clusters)) === 1
            @test name(get_node_local_marginals(clusters)[1]) === (:a, :b, :c)
            @test getfactorization(clusters) === ((1, 2, 3),)
            @test getfactorization(clusters, 1) === (1, 2, 3)
        end

        @testset let clusters = FactorNodeLocalClusters(
                interfaces, ((1, 2), (3,))
            )
            @test length(get_node_local_marginals(clusters)) === 2
            @test name(get_node_local_marginals(clusters)[1]) === (:a, :b)
            @test name(get_node_local_marginals(clusters)[2]) === :c
            @test getfactorization(clusters) === ((1, 2), (3,))
            @test getfactorization(clusters, 1) === (1, 2)
            @test getfactorization(clusters, 2) === (3,)
        end

        @testset let clusters = FactorNodeLocalClusters(
                interfaces, ((1,), (2, 3))
            )
            @test length(get_node_local_marginals(clusters)) === 2
            @test name(get_node_local_marginals(clusters)[1]) === :a
            @test name(get_node_local_marginals(clusters)[2]) === (:b, :c)
            @test getfactorization(clusters) === ((1,), (2, 3))
            @test getfactorization(clusters, 1) === (1,)
            @test getfactorization(clusters, 2) === (2, 3)
        end

        @testset let clusters = FactorNodeLocalClusters(
                interfaces, ((1,), (2,), (3,))
            )
            @test length(get_node_local_marginals(clusters)) === 3
            @test name(get_node_local_marginals(clusters)[1]) === :a
            @test name(get_node_local_marginals(clusters)[2]) === :b
            @test name(get_node_local_marginals(clusters)[3]) === :c
            @test getfactorization(clusters) === ((1,), (2,), (3,))
            @test getfactorization(clusters, 1) === (1,)
            @test getfactorization(clusters, 2) === (2,)
            @test getfactorization(clusters, 3) === (3,)
        end
    end
end

@testitem "clusterindex" tags = [:nodes] begin
    import ReactiveMP: FactorNodeLocalClusters, clusterindex

    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 2, 3),)), 1) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 2, 3),)), 2) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 2, 3),)), 3) === 1

    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 2), (3,))), 1) ===
        1
    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 2), (3,))), 2) ===
        1
    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 2), (3,))), 3) ===
        2

    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 3), (2,))), 1) ===
        1
    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 3), (2,))), 2) ===
        2
    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 3), (2,))), 3) ===
        1

    @test clusterindex(FactorNodeLocalClusters(missing, ((1,), (2, 3))), 1) ===
        1
    @test clusterindex(FactorNodeLocalClusters(missing, ((1,), (2, 3))), 2) ===
        2
    @test clusterindex(FactorNodeLocalClusters(missing, ((1,), (2, 3))), 3) ===
        2

    @test clusterindex(
        FactorNodeLocalClusters(missing, ((1,), (2,), (3,))), 1
    ) === 1
    @test clusterindex(
        FactorNodeLocalClusters(missing, ((1,), (2,), (3,))), 2
    ) === 2
    @test clusterindex(
        FactorNodeLocalClusters(missing, ((1,), (2,), (3,))), 3
    ) === 3

    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2, 3)]), 1) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2, 3)]), 2) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2, 3)]), 3) === 1

    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2), (3,)]), 1) ===
        1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2), (3,)]), 2) ===
        1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2), (3,)]), 3) ===
        2

    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 3), (2,)]), 1) ===
        1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 3), (2,)]), 2) ===
        2
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 3), (2,)]), 3) ===
        1

    @test clusterindex(FactorNodeLocalClusters(missing, [(1,), (2, 3)]), 1) ===
        1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1,), (2, 3)]), 2) ===
        2
    @test clusterindex(FactorNodeLocalClusters(missing, [(1,), (2, 3)]), 3) ===
        2

    @test clusterindex(
        FactorNodeLocalClusters(missing, [(1,), (2,), (3,)]), 1
    ) === 1
    @test clusterindex(
        FactorNodeLocalClusters(missing, [(1,), (2,), (3,)]), 2
    ) === 2
    @test clusterindex(
        FactorNodeLocalClusters(missing, [(1,), (2,), (3,)]), 3
    ) === 3
end

@testitem "clusterkey" tags = [:nodes] begin
    import ReactiveMP: NodeInterface, clusterkey

    interfaces = (NodeInterface(:a, randomvar()), NodeInterface(:b, randomvar()), NodeInterface(:c, randomvar()))

    @test clusterkey((1,), interfaces) === :a
    @test clusterkey((2,), interfaces) === :b
    @test clusterkey((3,), interfaces) === :c
    @test clusterkey((1, 2), interfaces) === (:a, :b)
    @test clusterkey((1, 3), interfaces) === (:a, :c)
    @test clusterkey((2, 3), interfaces) === (:b, :c)
    @test clusterkey((1, 2, 3), interfaces) === (:a, :b, :c)
end

@testitem "Correct initialization of clusters" tags = [:nodes] begin
    import ReactiveMP:
        FactorNodeActivationOptions, getlocalclusters, initialize_clusters!, getdata,
        get_node_local_marginals, get_stream_of_marginals
    using BayesBase, MessagePassingRulesBase

    include("../testutilities.jl")

    struct ArbitraryNode end

    @define_factor_node(node = ArbitraryNode, type = Stochastic, interfaces = [:out, :a, :b])

    @define_marginal_update_rule(
        node = ArbitraryNode, target = (:out, :a, :b),
        args = (m[:out]::PointMass, m[:a]::PointMass, m[:b]::PointMass),
        body = (args) -> PointMass(mean(args.m[:out]) + mean(args.m[:a]) + mean(args.m[:b])),
    )
    @define_marginal_update_rule(
        node = ArbitraryNode, target = (:out, :a),
        args = (m[:out]::PointMass, m[:a]::PointMass, q[:b]::PointMass),
        body = (args) -> PointMass(mean(args.m[:out]) + mean(args.m[:a]) - mean(args.q[:b])),
    )
    @define_marginal_update_rule(
        node = ArbitraryNode, target = (:out, :b),
        args = (m[:out]::PointMass, q[:a]::PointMass, m[:b]::PointMass),
        body = (args) -> PointMass(mean(args.m[:out]) + mean(args.m[:b]) - mean(args.q[:a])),
    )
    @define_marginal_update_rule(
        node = ArbitraryNode, target = (:a, :b),
        args = (q[:out]::PointMass, m[:a]::PointMass, m[:b]::PointMass),
        body = (args) -> PointMass(mean(args.m[:a]) + mean(args.m[:b]) - mean(args.q[:out])),
    )

    local_marginal_stream(node, i) = get_stream_of_marginals(get_node_local_marginals(getlocalclusters(node))[i])

    function initialized(factorisation, vout, va, vb)
        out, a, b = constvar(vout), constvar(va), constvar(vb)
        node = factornode(ArbitraryNode, [(:out, out), (:a, a), (:b, b)], factorisation)
        initialize_clusters!(getlocalclusters(node), node, FactorNodeActivationOptions())
        return node, (out, a, b)
    end

    @testset "Structured" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            node, _ = initialized(((:out, :a, :b),), vout, va, vb)
            @test length(get_node_local_marginals(getlocalclusters(node))) === 1
            @test PointMass(vout + va + vb) == getdata(check_stream_updated_once(local_marginal_stream(node, 1)))
        end
    end

    @testset "Factorized structured q(out, a)q(b)" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            node, (out, a, b) = initialized(((:out, :a), (:b,)), vout, va, vb)
            @test length(get_node_local_marginals(getlocalclusters(node))) === 2
            @test PointMass(vout + va - vb) == getdata(check_stream_updated_once(local_marginal_stream(node, 1)))
            @test local_marginal_stream(node, 2) === get_stream_of_marginals(b)
        end
    end

    @testset "Factorized structured q(out, b)q(a)" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            node, (out, a, b) = initialized(((:out, :b), (:a,)), vout, va, vb)
            @test length(get_node_local_marginals(getlocalclusters(node))) === 2
            @test PointMass(vout + vb - va) == getdata(check_stream_updated_once(local_marginal_stream(node, 1)))
            @test local_marginal_stream(node, 2) === get_stream_of_marginals(a)
        end
    end

    @testset "Factorized structured q(out)q(a, b)" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            node, (out, a, b) = initialized(((:out,), (:a, :b)), vout, va, vb)
            @test length(get_node_local_marginals(getlocalclusters(node))) === 2
            @test PointMass(va + vb - vout) == getdata(check_stream_updated_once(local_marginal_stream(node, 2)))
            @test local_marginal_stream(node, 1) === get_stream_of_marginals(out)
        end
    end
end
