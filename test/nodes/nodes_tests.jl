@testitem "GenericFactorNode constructor" begin
    import ReactiveMP: functionalform, getinterfaces, getinterface, name

    struct ArbitraryNodeType end

    @node ArbitraryNodeType Stochastic [a, b, c]

    function foo end

    @node typeof(foo) Deterministic [a, b, c]

    a = randomvar()
    b = datavar()
    c = constvar(1)

    @testset "functionalform" begin
        @test @inferred(functionalform(factornode(ArbitraryNodeType, [(:a, a), (:b, b), (:c, c)], ((1, 2, 3),)))) === ArbitraryNodeType
        @test @inferred(functionalform(factornode(foo, [(:a, a), (:b, b), (:c, c)], ((1, 2, 3),)))) === foo
    end

    @testset "getinterfaces" for fform in (ArbitraryNodeType, foo)
        let node = factornode(fform, [(:a, a), (:b, b), (:c, c)], ((1, 2, 3),))
            @test name.(getinterfaces(node)) == [:a, :b, :c]
            @test name(getinterface(node, 1)) == :a
            @test name(getinterface(node, 2)) == :b
            @test name(getinterface(node, 3)) == :c
        end
    end
end

@testitem "Predefined nodes should check the arguments supplied" begin
    struct StochasticNodeWithThreeArguments end
    struct DeterministicNodeWithFourArguments end

    @node StochasticNodeWithThreeArguments Stochastic [out, x, y, z]
    @node DeterministicNodeWithFourArguments Deterministic [out, x, y, z, w]

    out = randomvar()
    x = randomvar()
    y = randomvar()
    z = randomvar()
    w = randomvar()

    @test factornode(StochasticNodeWithThreeArguments, [(:out, out), (:x, x), (:y, y), (:z, z)], ((1, 2, 3),)) isa ReactiveMP.FactorNode
    @test factornode(DeterministicNodeWithFourArguments, [(:out, out), (:x, x), (:y, y), (:z, z), (:w, w)], ((1, 2, 3, 4),)) isa ReactiveMP.FactorNode

    @test_throws r"At least one argument is required for a factor node. Got none for `.*StochasticNodeWithThreeArguments`" factornode(StochasticNodeWithThreeArguments, [], ())
    @test_throws r"At least one argument is required for a factor node. Got none for `.*DeterministicNodeWithFourArguments`" factornode(DeterministicNodeWithFourArguments, [], ())
    @test_throws r"Expected 3 input arguments for `.*StochasticNodeWithThreeArguments`, got 1: x" factornode(StochasticNodeWithThreeArguments, [(:out, out), (:x, x)], ((1,),))
    @test_throws r"Expected 3 input arguments for `.*StochasticNodeWithThreeArguments`, got 2: x, y" factornode(
        StochasticNodeWithThreeArguments, [(:out, out), (:x, x), (:y, y)], ((1, 2),)
    )
    @test_throws r"Expected 3 input arguments for `.*StochasticNodeWithThreeArguments`, got 4: x, y, z, w" factornode(
        StochasticNodeWithThreeArguments, [(:out, out), (:x, x), (:y, y), (:z, z), (:w, w)], ((1, 2, 3, 4),)
    )
    @test_throws r"Expected 4 input arguments for `.*DeterministicNodeWithFourArguments`, got 1: x" factornode(DeterministicNodeWithFourArguments, [(:out, out), (:x, x)], ((1,),))
    @test_throws r"Expected 4 input arguments for `.*DeterministicNodeWithFourArguments`, got 2: x, y" factornode(
        DeterministicNodeWithFourArguments, [(:out, out), (:x, x), (:y, y)], ((1, 2),)
    )
    @test_throws r"Expected 4 input arguments for `.*DeterministicNodeWithFourArguments`, got 3: x, y, z" factornode(
        DeterministicNodeWithFourArguments, [(:out, out), (:x, x), (:y, y), (:z, z)], ((1, 2, 3),)
    )

    @test_throws r"`.*StochasticNodeWithThreeArguments` has duplicate entry for interface `w`. Did you pass an array \(e.g. `x`\) instead of an array element \(e\.g\. `x\[i\]`\)\? Check your variable indices\." factornode(
        StochasticNodeWithThreeArguments, [(:out, out), (:x, x), (:w, w), (:w, w)], ((1, 2, 3, 4),)
    )
    @test_throws r"`.*StochasticNodeWithThreeArguments` has duplicate entry for interface `w`. Did you pass an array \(e.g. `x`\) instead of an array element \(e\.g\. `x\[i\]`\)\? Check your variable indices\." factornode(
        StochasticNodeWithThreeArguments, [(:out, out), (:x, x), (:y, y), (:z, z), (:w, w), (:w, w)], ((1, 2, 3, 4, 5, 6),)
    )
end

@testitem "Generic node construction checks should not allocate" begin
    import ReactiveMP: prepare_interfaces_check_adjacent_duplicates, prepare_interfaces_check_nonempty, prepare_interfaces_check_numarguments

    struct NodeForCheckDuplicatesTest end
    @node NodeForCheckDuplicatesTest Stochastic [out, x, y, z]

    out = randomvar()
    x = randomvar()
    y = randomvar()
    z = randomvar()

    interfaces = [(:out, out), (:x, x), (:y, y), (:z, z)]
    # compile first
    function foo(interfaces)
        prepare_interfaces_check_nonempty(NodeForCheckDuplicatesTest, interfaces)
        prepare_interfaces_check_adjacent_duplicates(NodeForCheckDuplicatesTest, interfaces)
        prepare_interfaces_check_numarguments(NodeForCheckDuplicatesTest, interfaces)
    end
    foo(interfaces)
    @test (@allocated(foo(interfaces)) == 0)
    @test (@allocations(foo(interfaces)) == 0)
end

@testitem "`factornode` should throw an error if the functional form is not defined with the `@node` macro" begin
    struct UnknownDistribution end

    out = randomvar()
    alpha = randomvar()
    beta = randomvar()

    interfaces = [(:out, out), (:alpha, alpha), (:beta, beta)]

    @test_throws r"`.*UnknownDistribution.*` has been used but the `ReactiveMP` backend does not support `.*UnknownDistribution.*` as a factor node." factornode(
        UnknownDistribution, interfaces, ((1, 2, 3),)
    )
end
