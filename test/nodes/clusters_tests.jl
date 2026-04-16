@testitem "FactorNodeLocalMarginal" begin
    import Rocket: of, subscribe!, unsubscribe!
    import ReactiveMP:
        FactorNodeLocalMarginal,
        MarginalObservable,
        get_stream_of_marginals,
        set_stream_of_marginals!,
        tag,
        name

    @testset let localmarginal = FactorNodeLocalMarginal(:a)
        @test name(localmarginal) === :a
        @test tag(localmarginal) === Val{:a}()
        @test occursin("a", repr(localmarginal))
        # The stream is not set
        @test_throws UndefRefError get_stream_of_marginals(localmarginal)

        m = MarginalObservable()

        set_stream_of_marginals!(localmarginal, m)

        @test get_stream_of_marginals(localmarginal) === m
    end

    @testset let localmarginal = FactorNodeLocalMarginal(:b)
        @test name(localmarginal) === :b
        @test tag(localmarginal) === Val{:b}()
        @test occursin("b", repr(localmarginal))
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

@testitem "FactorNodeLocalClusters constructor" begin
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
            @test name(get_node_local_marginals(clusters)[1]) === :a_b_c
            @test getfactorization(clusters) === ((1, 2, 3),)
            @test getfactorization(clusters, 1) === (1, 2, 3)
        end

        @testset let clusters = FactorNodeLocalClusters(
                interfaces, ((1, 2), (3,))
            )
            @test length(get_node_local_marginals(clusters)) === 2
            @test name(get_node_local_marginals(clusters)[1]) === :a_b
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
            @test name(get_node_local_marginals(clusters)[2]) === :b_c
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

@testitem "clusterindex" begin
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

@testitem "clustername" begin
    import ReactiveMP: NodeInterface, FactorNodeLocalClusters, clustername

    a = NodeInterface(:a, randomvar())
    b = NodeInterface(:b, randomvar())
    c = NodeInterface(:c, randomvar())

    interfaces = (a, b, c)

    @test clustername(interfaces) === :a_b_c
    @test clustername([a, b]) === :a_b
    @test clustername([b, c]) === :b_c
    @test clustername([a, c]) === :a_c
    @test clustername((a, b)) === :a_b
    @test clustername((b, c)) === :b_c
    @test clustername((a, c)) === :a_c

    @test clustername((1,), interfaces) === :a
    @test clustername((2,), interfaces) === :b
    @test clustername((3,), interfaces) === :c
    @test clustername((1, 2), interfaces) === :a_b
    @test clustername((1, 3), interfaces) === :a_c
    @test clustername((2, 3), interfaces) === :b_c
    @test clustername((1, 2, 3), interfaces) === :a_b_c
end

@testitem "Correct initialization of clusters" begin
    import ReactiveMP:
        NodeInterface,
        FactorNodeLocalClusters,
        clustername,
        FactorNodeActivationOptions,
        RandomVariableActivationOptions,
        activate!,
        getlocalclusters,
        initialize_clusters!,
        getdata,
        default_functional_dependencies,
        get_node_local_marginals,
        get_stream_of_marginals

    using BayesBase

    include("../testutilities.jl")

    struct ArbitraryNode end

    @node ArbitraryNode Stochastic [out, a, b]

    @marginalrule ArbitraryNode(:out_a_b) (
        m_out::PointMass, m_a::PointMass, m_b::PointMass
    ) = begin
        return PointMass(mean(m_out) + mean(m_a) + mean(m_b))
    end

    @marginalrule ArbitraryNode(:out_a) (
        m_out::PointMass, m_a::PointMass, q_b::PointMass
    ) = begin
        return PointMass(mean(m_out) + mean(m_a) - mean(q_b))
    end

    @marginalrule ArbitraryNode(:out_b) (
        m_out::PointMass, q_a::PointMass, m_b::PointMass
    ) = begin
        return PointMass(mean(m_out) + mean(m_b) - mean(q_a))
    end

    @marginalrule ArbitraryNode(:a_b) (
        q_out::PointMass, m_a::PointMass, m_b::PointMass
    ) = begin
        return PointMass(mean(m_a) + mean(m_b) - mean(q_out))
    end

    dependencies = default_functional_dependencies(ArbitraryNode)

    @testset "Structured" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            out = constvar(vout)
            a = constvar(va)
            b = constvar(vb)

            node = factornode(
                ArbitraryNode, [(:out, out), (:a, a), (:b, b)], ((1, 2, 3),)
            )

            options = FactorNodeActivationOptions(
                nothing, nothing, nothing, nothing, nothing, nothing
            )

            @test length(get_node_local_marginals(getlocalclusters(node))) === 1

            initialize_clusters!(
                getlocalclusters(node), dependencies, node, options
            )

            @test PointMass(vout + va + vb) == getdata(
                check_stream_updated_once(
                    get_stream_of_marginals(
                        get_node_local_marginals(getlocalclusters(node))[1]
                    ),
                ),
            )
        end
    end

    @testset "Factorized structured q(out, a)q(b)" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            out = constvar(vout)
            a = constvar(va)
            b = constvar(vb)

            node = factornode(
                ArbitraryNode, [(:out, out), (:a, a), (:b, b)], ((1, 2), (3,))
            )

            options = FactorNodeActivationOptions(
                nothing, nothing, nothing, nothing, nothing, nothing
            )

            @test length(get_node_local_marginals(getlocalclusters(node))) === 2

            initialize_clusters!(
                getlocalclusters(node), dependencies, node, options
            )

            @test PointMass(vout + va - vb) == getdata(
                check_stream_updated_once(
                    get_stream_of_marginals(
                        get_node_local_marginals(getlocalclusters(node))[1]
                    ),
                ),
            )
            @test get_stream_of_marginals(
                get_node_local_marginals(getlocalclusters(node))[2]
            ) === get_stream_of_marginals(b)
        end
    end

    @testset "Factorized structured q(out, b)q(a)" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            out = constvar(vout)
            a = constvar(va)
            b = constvar(vb)

            node = factornode(
                ArbitraryNode, [(:out, out), (:a, a), (:b, b)], ((1, 3), (2,))
            )

            options = FactorNodeActivationOptions(
                nothing, nothing, nothing, nothing, nothing, nothing
            )

            @test length(get_node_local_marginals(getlocalclusters(node))) === 2

            initialize_clusters!(
                getlocalclusters(node), dependencies, node, options
            )

            @test PointMass(vout + vb - va) == getdata(
                check_stream_updated_once(
                    get_stream_of_marginals(
                        get_node_local_marginals(getlocalclusters(node))[1]
                    ),
                ),
            )
            @test get_stream_of_marginals(
                get_node_local_marginals(getlocalclusters(node))[2]
            ) === get_stream_of_marginals(a)
        end
    end

    @testset "Factorized structured q(out)q(a, b)" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            out = constvar(vout)
            a = constvar(va)
            b = constvar(vb)

            node = factornode(
                ArbitraryNode, [(:out, out), (:a, a), (:b, b)], ((1,), (2, 3))
            )

            options = FactorNodeActivationOptions(
                nothing, nothing, nothing, nothing, nothing, nothing
            )

            @test length(get_node_local_marginals(getlocalclusters(node))) === 2

            initialize_clusters!(
                getlocalclusters(node), dependencies, node, options
            )

            @test PointMass(va + vb - vout) == getdata(
                check_stream_updated_once(
                    get_stream_of_marginals(
                        get_node_local_marginals(getlocalclusters(node))[2]
                    ),
                ),
            )
            @test get_stream_of_marginals(
                get_node_local_marginals(getlocalclusters(node))[1]
            ) === get_stream_of_marginals(out)
        end
    end
end
