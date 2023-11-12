module ContinuousTransitionNodeTest

using Test, ReactiveMP, Random, Distributions, BayesBase, ExponentialFamily

import ReactiveMP: getdimensionality, getmasks, ctcompanion_matrix, getunits

@testset "ContinuousTransitionNode" begin
    @testset "Creation" begin
        meta = CTMeta((a) -> reshape(a, 2, 3), 6)  # Example transformation function and vector length
        node = make_node(ContinuousTransition, FactorNodeCreationOptions(nothing, meta, nothing))

        @test functionalform(node) === ContinuousTransition
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:y, :x, :a, :W)
        @test factorisation(node) === ((1, 2, 3, 4),)
        @test getdimensionality(metadata(node)) == (2, 3)  # Based on the transformation function dimensions
    end

    @testset "AverageEnergy" begin
        # This is an example setup, you'll need to adjust the distributions and marginals according to your needs
        q_y_x = MvNormalMeanCovariance(zeros(5), diageye(5))
        q_a = MvNormalMeanCovariance(zeros(6), diageye(6))  # Adjust the dimension according to `a`
        q_W = Wishart(3, diageye(2))  # Adjust the degrees of freedom and scale matrix as needed

        marginals = (Marginal(q_y_x, false, false, nothing), Marginal(q_a, false, false, nothing), Marginal(q_W, false, false, nothing))
        meta = CTMeta((a) -> reshape(a, 2, 3), 6)

        @test score(AverageEnergy(), ContinuousTransition, Val{(:y_x, :a, :W)}(), marginals, meta) ≈ 13.415092731310878 #ExpectedValue
    end

    @testset "CTransitionFunctionality" begin
        a = rand(6)  # Example vector `a`
        meta = CTMeta((a) -> reshape(a, 2, 3), 6)
        A = ctcompanion_matrix(a, meta)

        @test size(A) == (2, 3)
        @test A == reshape(a, 2, 3)  # This is based on the transformation function provided in meta
    end

    @testset "MetadataFunctionality" begin
        meta = CTMeta((a) -> reshape(a, 2, 3), 6)

        @test getdimensionality(meta) == (2, 3)
        @test length(getmasks(meta, rand(6))) == 2  # Based on `dy`
        @test length(getunits(meta)) == 2  # Based on `dy`
    end
end

end
