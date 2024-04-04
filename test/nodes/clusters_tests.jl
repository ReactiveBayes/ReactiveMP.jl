@testitem "FactorNodeLocalMarginal" begin
    import ReactiveMP: FactorNodeLocalMarginal, MarginalObservable, getmarginal, setmarginal!, tag, name

    @testset let localmarginal = FactorNodeLocalMarginal(:a)
        @test name(localmarginal) === :a
        @test tag(localmarginal) === Val{:a}()
        @test occursin("a", repr(localmarginal))
        # The stream is not set
        @test_throws UndefRefError getmarginal(localmarginal)

        m = MarginalObservable()

        setmarginal!(localmarginal, m)

        @test getmarginal(localmarginal) === m
    end
end

@testitem "FactorNodeLocalClusters constructor" begin
    import ReactiveMP: NodeInterface, FactorNodeLocalClusters, getfactorization, getmarginals, getmarginal, name

    a = NodeInterface(:a, randomvar())
    b = NodeInterface(:b, randomvar())
    c = NodeInterface(:c, randomvar())

    # Interfaces can be both tuples and arrays
    for interfaces in [(a, b, c), [a, b, c]]
        @testset let clusters = FactorNodeLocalClusters(interfaces, ((1, 2, 3),))
            @test length(getmarginals(clusters)) === 1
            @test name(getmarginal(clusters, 1)) === :a_b_c
            @test getfactorization(clusters) === ((1, 2, 3),)
            @test getfactorization(clusters, 1) === (1, 2, 3)
        end

        @testset let clusters = FactorNodeLocalClusters(interfaces, ((1, 2), (3,)))
            @test length(getmarginals(clusters)) === 2
            @test name(getmarginal(clusters, 1)) === :a_b
            @test name(getmarginal(clusters, 2)) === :c
            @test getfactorization(clusters) === ((1, 2), (3,))
            @test getfactorization(clusters, 1) === (1, 2)
            @test getfactorization(clusters, 2) === (3,)
        end

        @testset let clusters = FactorNodeLocalClusters(interfaces, ((1,), (2, 3)))
            @test length(getmarginals(clusters)) === 2
            @test name(getmarginal(clusters, 1)) === :a
            @test name(getmarginal(clusters, 2)) === :b_c
            @test getfactorization(clusters) === ((1,), (2, 3))
            @test getfactorization(clusters, 1) === (1,)
            @test getfactorization(clusters, 2) === (2, 3)
        end

        @testset let clusters = FactorNodeLocalClusters(interfaces, ((1,), (2,), (3,)))
            @test length(getmarginals(clusters)) === 3
            @test name(getmarginal(clusters, 1)) === :a
            @test name(getmarginal(clusters, 2)) === :b
            @test name(getmarginal(clusters, 3)) === :c
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

    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 2), (3,))), 1) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 2), (3,))), 2) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 2), (3,))), 3) === 2

    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 3), (2,))), 1) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 3), (2,))), 2) === 2
    @test clusterindex(FactorNodeLocalClusters(missing, ((1, 3), (2,))), 3) === 1

    @test clusterindex(FactorNodeLocalClusters(missing, ((1,), (2, 3))), 1) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, ((1,), (2, 3))), 2) === 2
    @test clusterindex(FactorNodeLocalClusters(missing, ((1,), (2, 3))), 3) === 2

    @test clusterindex(FactorNodeLocalClusters(missing, ((1,), (2,), (3,))), 1) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, ((1,), (2,), (3,))), 2) === 2
    @test clusterindex(FactorNodeLocalClusters(missing, ((1,), (2,), (3,))), 3) === 3

    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2, 3)]), 1) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2, 3)]), 2) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2, 3)]), 3) === 1

    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2), (3,)]), 1) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2), (3,)]), 2) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 2), (3,)]), 3) === 2

    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 3), (2,)]), 1) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 3), (2,)]), 2) === 2
    @test clusterindex(FactorNodeLocalClusters(missing, [(1, 3), (2,)]), 3) === 1

    @test clusterindex(FactorNodeLocalClusters(missing, [(1,), (2, 3)]), 1) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1,), (2, 3)]), 2) === 2
    @test clusterindex(FactorNodeLocalClusters(missing, [(1,), (2, 3)]), 3) === 2

    @test clusterindex(FactorNodeLocalClusters(missing, [(1,), (2,), (3,)]), 1) === 1
    @test clusterindex(FactorNodeLocalClusters(missing, [(1,), (2,), (3,)]), 2) === 2
    @test clusterindex(FactorNodeLocalClusters(missing, [(1,), (2,), (3,)]), 3) === 3
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
        default_functional_dependencies

    using BayesBase

    include("../testutilities.jl")

    struct ArbitraryNode end

    @node ArbitraryNode Stochastic [out, a, b]

    @marginalrule ArbitraryNode(:out_a_b) (m_out::PointMass, m_a::PointMass, m_b::PointMass) = begin
        return PointMass(mean(m_out) + mean(m_a) + mean(m_b))
    end

    @marginalrule ArbitraryNode(:out_a) (m_out::PointMass, m_a::PointMass, q_b::PointMass) = begin
        return PointMass(mean(m_out) + mean(m_a) - mean(q_b))
    end

    @marginalrule ArbitraryNode(:out_b) (m_out::PointMass, q_a::PointMass, m_b::PointMass) = begin
        return PointMass(mean(m_out) + mean(m_b) - mean(q_a))
    end

    @marginalrule ArbitraryNode(:a_b) (q_out::PointMass, m_a::PointMass, m_b::PointMass) = begin
        return PointMass(mean(m_a) + mean(m_b) - mean(q_out))
    end

    dependencies = default_functional_dependencies(ArbitraryNode)

    @testset "Structured" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            out = constvar(vout)
            a = constvar(va)
            b = constvar(vb)

            node = factornode(ArbitraryNode, [(:out, out), (:a, a), (:b, b)], ((1, 2, 3),))

            options = FactorNodeActivationOptions(nothing, nothing, nothing, nothing, nothing)

            @test length(getmarginals(getlocalclusters(node))) === 1

            initialize_clusters!(getlocalclusters(node), dependencies, node, options)

            @test PointMass(vout + va + vb) == getdata(check_stream_updated_once(getmarginal(getmarginal(getlocalclusters(node), 1))))
        end
    end

    @testset "Factorized structured q(out, a)q(b)" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            out = constvar(vout)
            a = constvar(va)
            b = constvar(vb)

            node = factornode(ArbitraryNode, [(:out, out), (:a, a), (:b, b)], ((1, 2), (3,)))

            options = FactorNodeActivationOptions(nothing, nothing, nothing, nothing, nothing)

            @test length(getmarginals(getlocalclusters(node))) === 2

            initialize_clusters!(getlocalclusters(node), dependencies, node, options)

            @test PointMass(vout + va - vb) == getdata(check_stream_updated_once(getmarginal(getmarginal(getlocalclusters(node), 1))))
            @test getmarginal(getmarginal(getlocalclusters(node), 2)) === getmarginal(b, IncludeAll())
        end
    end

    @testset "Factorized structured q(out, b)q(a)" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            out = constvar(vout)
            a = constvar(va)
            b = constvar(vb)

            node = factornode(ArbitraryNode, [(:out, out), (:a, a), (:b, b)], ((1, 3), (2,)))

            options = FactorNodeActivationOptions(nothing, nothing, nothing, nothing, nothing)

            @test length(getmarginals(getlocalclusters(node))) === 2

            initialize_clusters!(getlocalclusters(node), dependencies, node, options)

            @test PointMass(vout + vb - va) == getdata(check_stream_updated_once(getmarginal(getmarginal(getlocalclusters(node), 1))))
            @test getmarginal(getmarginal(getlocalclusters(node), 2)) === getmarginal(a, IncludeAll())
        end
    end

    @testset "Factorized structured q(out)q(a, b)" begin
        for (vout, va, vb) in [rand(3) for _ in 1:5]
            out = constvar(vout)
            a = constvar(va)
            b = constvar(vb)

            node = factornode(ArbitraryNode, [(:out, out), (:a, a), (:b, b)], ((1,), (2, 3)))

            options = FactorNodeActivationOptions(nothing, nothing, nothing, nothing, nothing)

            @test length(getmarginals(getlocalclusters(node))) === 2

            initialize_clusters!(getlocalclusters(node), dependencies, node, options)

            @test PointMass(va + vb - vout) == getdata(check_stream_updated_once(getmarginal(getmarginal(getlocalclusters(node), 2))))
            @test getmarginal(getmarginal(getlocalclusters(node), 1)) === getmarginal(out, IncludeAll())
        end
    end
end
