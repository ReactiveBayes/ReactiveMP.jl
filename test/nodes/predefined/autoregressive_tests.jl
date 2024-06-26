
@testitem "AutoregressiveNode" begin
    using ReactiveMP, Random, Distributions, BayesBase, ExponentialFamily

    import ReactiveMP: getvform, getorder, getstype

    @testset "AverageEnergy" begin
        q_y = NormalMeanVariance(0.0, 1.0)
        q_x = NormalMeanVariance(0.0, 1.0)
        q_θ = NormalMeanVariance(0.0, 1.0)
        q_γ = GammaShapeRate(2.0, 3.0)

        marginals = (Marginal(q_y, false, false, nothing), Marginal(q_x, false, false, nothing), Marginal(q_θ, false, false, nothing), Marginal(q_γ, false, false, nothing))
        @test score(AverageEnergy(), Autoregressive, Val{(:y, :x, :θ, :γ)}(), marginals, ARMeta(Univariate, 1, ARsafe())) ≈ 1.92351917665

        q_y = MvNormalMeanCovariance(zeros(2), diageye(2))
        q_x = MvNormalMeanCovariance(zeros(2), diageye(2))
        q_θ = MvNormalMeanCovariance(zeros(2), diageye(2))
        q_γ = GammaShapeRate(2.0, 3.0)

        marginals = (Marginal(q_y, false, false, nothing), Marginal(q_x, false, false, nothing), Marginal(q_θ, false, false, nothing), Marginal(q_γ, false, false, nothing))
        @test score(AverageEnergy(), Autoregressive, Val{(:y, :x, :θ, :γ)}(), marginals, ARMeta(Univariate, 1, ARsafe())) ≈ 2.25685250999

        q_y_x = MvNormalMeanCovariance(zeros(2), diageye(2))
        q_θ = NormalMeanVariance(0.0, 1.0)
        q_γ = GammaShapeRate(2.0, 3.0)

        marginals = (Marginal(q_y_x, false, false, nothing), Marginal(q_θ, false, false, nothing), Marginal(q_γ, false, false, nothing))
        @test score(AverageEnergy(), Autoregressive, Val{(:y_x, :θ, :γ)}(), marginals, ARMeta(Univariate, 1, ARsafe())) ≈ 1.92351917665616
    end

    @testset "ARTransitionMatrix" begin
        rng = MersenneTwister(1233)

        for γ in randn(rng, 3), order in 2:4
            transition = ReactiveMP.ARTransitionMatrix(order, γ)
            matrix = rand(rng, order, order)
            ftransition = zeros(order, order)
            ftransition[1] = inv(γ)

            @test broadcast(+, matrix, transition) == (matrix + ftransition)
            @test_throws DimensionMismatch broadcast(+, zeros(order + 1, order + 1), transition)

            @test ReactiveMP.add_transition(matrix, transition) == (matrix + ftransition)
            @test_throws DimensionMismatch ReactiveMP.add_transition(zeros(order + 1, order + 1), transition)

            cmatrix = copy(matrix)
            broadcast!(+, cmatrix, transition)
            @test cmatrix == (matrix + ftransition)

            cmatrix = copy(matrix)
            ReactiveMP.add_transition!(cmatrix, transition)
            @test cmatrix == (matrix + ftransition)
        end
    end
end
