
@testitem "ContinuousTransitionNode" begin
    using Test, ReactiveMP, Random, Distributions, BayesBase, ExponentialFamily

    import ReactiveMP: getjacobians, gettransformation, ctcompanion_matrix
    # TODO: A more rigorous test suit for the average energy of CTransition needs to be added 
    dy, dx = 2, 3
    meta = CTMeta(a -> reshape(a, dy, dx))

    @testset "AverageEnergy" begin
        q_y = MvNormalMeanCovariance(zeros(dy), diageye(dy))
        q_x = MvNormalMeanCovariance(zeros(dx), diageye(dx))

        q_y_x = MvNormalMeanCovariance([mean(q_y); mean(q_x)], [cov(q_y) zeros(dy, dx); zeros(dx, dy) cov(q_x)])
        q_a = MvNormalMeanCovariance(zeros(dx * dy), diageye(dx * dy))
        q_W = Wishart(dy + 1, diageye(dy))

        marginals_st = (Marginal(q_y_x, false, false, nothing), Marginal(q_a, false, false, nothing), Marginal(q_W, false, false, nothing))
        marginals_mf = (Marginal(q_y, false, false, nothing), Marginal(q_x, false, false, nothing), Marginal(q_a, false, false, nothing), Marginal(q_W, false, false, nothing))

        # 12,992 is a result of manual calculation 
        @test score(AverageEnergy(), ContinuousTransition, Val{(:y_x, :a, :W)}(), marginals_st, meta) ≈ 12.992 atol = 1e-2
        # 12,07336 is a result of manual calculation 
        @test score(AverageEnergy(), ContinuousTransition, Val{(:y, :x, :a, :W)}(), marginals_mf, meta) ≈ 12.07736 atol = 1e-2
    end

    @testset "ContinuousTransition Functionality" begin
        m_a = randn(6)
        A = ctcompanion_matrix(m_a, zeros(length(m_a)), meta)

        @test size(A) == (dy, dx)
        @test A == gettransformation(meta)(m_a)
    end
end
