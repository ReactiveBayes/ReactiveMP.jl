module ContinuousTransitionNodeTest

using Test, ReactiveMP, Random, Distributions, BayesBase, ExponentialFamily

import ReactiveMP: getdimensionality, getjacobians, gettransformation, getunits, ctcompanion_matrix

@testset "ContinuousTransitionNode" begin
    dy, dx = 2, 3
    a0 = rand(dx * dy)  # Example vector `a0`
    meta = CTMeta(a -> reshape(a, dy, dx), a0)
    @testset "Creation" begin
        node = make_node(ContinuousTransition, FactorNodeCreationOptions(nothing, meta, nothing))

        @test functionalform(node) === ContinuousTransition
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:y, :x, :a, :W)
        @test factorisation(node) === ((1, 2, 3, 4),)
        @test getdimensionality(metadata(node)) == (dy, dx)  # Based on the transformation function dimensions
    end

    @testset "AverageEnergy" begin
        # This is an example setup, you'll need to adjust the distributions and marginals according to your needs
        q_y_x = MvNormalMeanCovariance(zeros(5), diageye(5))
        q_a = MvNormalMeanCovariance(zeros(6), diageye(6))  # Adjust the dimension according to `a`
        q_W = Wishart(3, diageye(2))

        marginals = (Marginal(q_y_x, false, false, nothing), Marginal(q_a, false, false, nothing), Marginal(q_W, false, false, nothing))

        @test score(AverageEnergy(), ContinuousTransition, Val{(:y_x, :a, :W)}(), marginals, meta) ≈ 13.415092731310878
        @show getjacobians(meta, a0)
    end

    @testset "CTransition Functionality" begin
        A = ctcompanion_matrix(a0, zeros(length(a0)), meta)

        @test size(A) == (dy, dx)
        @test A == gettransformation(meta)(a0)
    end

    @testset "Metadata Functionality" begin
        @test getdimensionality(meta) == (dy, dx)
        @test length(getjacobians(meta, a0)) == dy  # Based on `dy`
        @test length(getunits(meta)) == dy  # Based on `dy`
    end
end

end
