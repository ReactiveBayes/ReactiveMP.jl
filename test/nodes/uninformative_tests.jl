@testitem "UninformativeNode" begin
    module UninformativeNodeTest

    using Test, ReactiveMP, Random, BayesBase, ExponentialFamily
    @testset "Creation" begin
        node = make_node(Uninformative)

        @test functionalform(node) === Uninformative
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out,)
        @test factorisation(node) === ((1,),)
        @test localmarginalnames(node) === (:out,)
        @test metadata(node) === nothing
    end

    @testset "Product must not be affected" begin
        @test prod(GenericProd(), Uninformative(), NormalMeanVariance(0, 1)) == NormalMeanVariance(0, 1)
        @test prod(GenericProd(), NormalMeanVariance(3, 4), Uninformative()) == NormalMeanVariance(3, 4)
        @test prod(GenericProd(), Uninformative(), Uninformative()) === Uninformative()
    end
    end
end
