
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

    # TODO: multivariate case coverage

end

@testitem "AutoregressiveNode: is_univariate, is_safe, is_unsafe, default_meta, is_multivariate cases, ar_unit, ARPrecisionMatrix" begin
    using ReactiveMP, Distributions, ExponentialFamily, LazyArrays, LinearAlgebra, Test
    import ReactiveMP: ARMeta, ARsafe, ARunsafe, AR, ar_unit, ar_precision, ARPrecisionMatrix
    @testset "ARMeta and ARPrecisionMatrix extensions" begin

        # --- ARMeta property tests ---
        meta_uni = ARMeta(Univariate, 1, ARsafe())
        meta_multi = ARMeta(Multivariate, 3, ARunsafe())

        @test ReactiveMP.is_univariate(meta_uni)
        @test !ReactiveMP.is_multivariate(meta_uni)
        @test ReactiveMP.is_safe(meta_uni)
        @test !ReactiveMP.is_unsafe(meta_uni)

        @test ReactiveMP.is_multivariate(meta_multi)
        @test !ReactiveMP.is_univariate(meta_multi)
        @test ReactiveMP.is_unsafe(meta_multi)
        @test !ReactiveMP.is_safe(meta_multi)

        # --- default_meta should throw ---
        @test_throws ErrorException ReactiveMP.default_meta(AR)

        # --- ar_unit coverage ---
        @test ar_unit(Univariate, 1) == 1.0
        @test ar_unit(Multivariate, 3)[1] ≈ 1.0
        @test eltype(ar_unit(Float32, Univariate, 1)) === Float32

        # --- ar_precision and ARPrecisionMatrix ---
        γ = 2.5
        order = 3
        pm = ar_precision(Multivariate, order, γ)
        @test size(pm) == (3, 3)
        @test pm[1, 1] == γ
        @test pm[2, 2] ≈ convert(Float64, huge)
        @test pm[1, 2] == 0.0

        # add_precision & add_precision!
        A = zeros(3, 3)
        B = copy(A)
        res = ReactiveMP.add_precision(A, pm)
        @test res[1, 1] == γ
        @test res[2, 2] ≈ huge

        ReactiveMP.add_precision!(B, pm)
        @test B[1, 1] == γ
        @test B[3, 3] ≈ huge

        # Real + Real overloads
        @test ReactiveMP.add_precision!(1.0, 2.0) == 3.0
        @test ReactiveMP.add_precision(1.0, 2.0) == 3.0

        # Broadcast! directly
        C = zeros(3, 3)
        broadcast!(+, C, pm)
        @test C[1, 1] == γ
        @test all(diag(C)[2:end] .≈ huge)
    end
end
