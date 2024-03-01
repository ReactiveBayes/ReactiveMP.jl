@testitem "GenericFactorNode constructor" begin
    import ReactiveMP: GenericFactorNode, functionalform, getinterfaces, getinterface

    struct ArbitraryNodeType end

    function foo end

    @testset "functionalform" begin
        @test @inferred(functionalform(GenericFactorNode(ArbitraryNodeType, (; )))) === ArbitraryNodeType
        @test @inferred(functionalform(GenericFactorNode(foo, (; )))) === typeof(foo)
    end

    @testset "getinterfaces" for fform in (ArbitraryNodeType, foo)
        a = RandomVariable()
        b = DataVariable()
        c = ConstVariable(1)

        let node = GenericFactorNode(fform, (a = a, b = b, c = c))
            @test name.(getinterfaces(node)) == (:a, :b, :c)
            @test name(getinterface(node, 1)) == :a
            @test name(getinterface(node, 2)) == :b
            @test name(getinterface(node, 3)) == :c
        end
    end
end