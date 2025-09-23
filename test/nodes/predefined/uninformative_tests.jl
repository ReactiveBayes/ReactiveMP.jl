
@testitem "UninformativeNode" begin
    using Test, ReactiveMP, Random, BayesBase, ExponentialFamily
    using BayesBase: TerminalProdArgument

    @testset "Product must not be affected" begin
        @test prod(GenericProd(), Uninformative(), NormalMeanVariance(0, 1)) == NormalMeanVariance(0, 1)
        @test prod(GenericProd(), NormalMeanVariance(3, 4), Uninformative()) == NormalMeanVariance(3, 4)
        @test prod(GenericProd(), Uninformative(), Uninformative()) === Uninformative()
    end

    @testset "Product with BayesBase.TerminalProdArgument" begin
        @test prod(GenericProd(), Uninformative(), TerminalProdArgument(NormalMeanVariance(0, 1))) == TerminalProdArgument(NormalMeanVariance(0, 1))
        @test prod(GenericProd(), TerminalProdArgument(NormalMeanVariance(3, 4)), Uninformative()) == TerminalProdArgument(NormalMeanVariance(3, 4))
        @test prod(GenericProd(), Uninformative(), TerminalProdArgument(PointMass(0))) === TerminalProdArgument(PointMass(0))
        @test prod(GenericProd(), TerminalProdArgument(PointMass(0)), Uninformative()) === TerminalProdArgument(PointMass(0))
    end
end
