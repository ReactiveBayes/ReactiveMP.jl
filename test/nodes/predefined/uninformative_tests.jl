
@testitem "UninformativeNode" begin
    using Test, ReactiveMP, Random, BayesBase, ExponentialFamily

    @testset "Product must not be affected" begin
        @test prod(GenericProd(), Uninformative(), NormalMeanVariance(0, 1)) == NormalMeanVariance(0, 1)
        @test prod(GenericProd(), NormalMeanVariance(3, 4), Uninformative()) == NormalMeanVariance(3, 4)
        @test prod(GenericProd(), Uninformative(), Uninformative()) === Uninformative()
    end
end
