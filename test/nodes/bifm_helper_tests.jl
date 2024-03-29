
@testitem "BIFMHelperNode" begin
    using ReactiveMP, Random, ExponentialFamily, BayesBase

    import ReactiveMP: @test_rules

    @testset "Creation" begin
        node = make_node(BIFMHelper)

        @test functionalform(node) === BIFMHelper
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :in)
        @test factorisation(node) === ((1, 2),)
    end

    @testset "Average energy" begin
        node = make_node(BIFMHelper)

        @test score(
            AverageEnergy(),
            BIFMHelper,
            Val{(:out, :in)}(),
            (Marginal(MvNormalMeanCovariance([1, 1], [2 0; 0 3]), false, false, nothing), Marginal(MvNormalMeanCovariance([1, 1], [2 0; 0 3]), false, false, nothing)),
            nothing
        ) ≈ entropy(MvNormalMeanCovariance([1, 1], [2 0; 0 3]))

        @test score(
            AverageEnergy(),
            BIFMHelper,
            Val{(:out, :in)}(),
            (Marginal(MvNormalMeanCovariance([1, 2], [2 0; 0 1]), false, false, nothing), Marginal(MvNormalMeanPrecision([1, 2], [0.5 0; 0 1]), false, false, nothing)),
            nothing
        ) ≈ entropy(MvNormalMeanCovariance([1, 2], [2 0; 0 1]))
    end
end
