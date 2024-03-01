@testitem "FactorNodeLocalMarginal" begin
    import ReactiveMP: FactorNodeLocalMarginal, MarginalObservable, getstream, setstream!, tag

    @testset let localmarginal = FactorNodeLocalMarginal(:a)
        @test name(localmarginal) === :a
        @test tag(localmarginal) === Val{:a}()
        @test occursin("a", repr(localmarginal))
        # The stream is not set
        @test_throws UndefRefError getstream(localmarginal)

        m = MarginalObservable()

        setstream!(localmarginal, m)

        @test getstream(localmarginal) === m
    end
end

@testitem "FactorNodeLocalClusters constructor" begin
    import ReactiveMP: NodeInterface, FactorNodeLocalClusters, getfactorization, getmarginals, getmarginal

    a = NodeInterface(:a, RandomVariable())
    b = NodeInterface(:b, RandomVariable())
    c = NodeInterface(:c, RandomVariable())

    interfaces = (a, b, c)

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

    a = NodeInterface(:a, RandomVariable())
    b = NodeInterface(:b, RandomVariable())
    c = NodeInterface(:c, RandomVariable())

    interfaces = (a, b, c)

    @test clustername((1,), interfaces) === :a
    @test clustername((2,), interfaces) === :b
    @test clustername((3,), interfaces) === :c
    @test clustername((1, 2), interfaces) === :a_b
    @test clustername((1, 3), interfaces) === :a_c
    @test clustername((2, 3), interfaces) === :b_c
    @test clustername((1, 2, 3), interfaces) === :a_b_c
end