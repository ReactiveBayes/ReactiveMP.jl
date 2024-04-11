
@testitem "NormalMixtureNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily

    import ReactiveMP: ManyOf
    import ExponentialFamily: WishartFast

    @testset "AverageEnergy" begin
        begin
            q_out = NormalMeanVariance(0.0, 1.0)
            q_switch = Bernoulli(0.2)
            q_m = (NormalMeanVariance(1.0, 2.0), NormalMeanVariance(3.0, 4.0))
            q_p = (GammaShapeRate(2.0, 3.0), GammaShapeRate(4.0, 5.0))

            marginals = (
                Marginal(q_out, false, false, nothing),
                Marginal(q_switch, false, false, nothing),
                ManyOf(map(q_m_ -> Marginal(q_m_, false, false, nothing), q_m)),
                ManyOf(map(q_p_ -> Marginal(q_p_, false, false, nothing), q_p))
            )

            ref_val =
                0.8 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[1], q_p[1])), nothing)) +
                0.2 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[2], q_p[2])), nothing))
            @test score(AverageEnergy(), NormalMixture, Val{(:out, :switch, :m, :p)}(), marginals, nothing) ≈ ref_val
        end

        begin
            q_out = NormalMeanVariance(1.0, 1.0)
            q_switch = Bernoulli(0.4)
            q_m = (NormalMeanVariance(3.0, 2.0), NormalMeanVariance(3.0, 4.0))
            q_p = (GammaShapeRate(2.0, 3.0), GammaShapeRate(1.0, 5.0))

            marginals = (
                Marginal(q_out, false, false, nothing),
                Marginal(q_switch, false, false, nothing),
                ManyOf(map(q_m_ -> Marginal(q_m_, false, false, nothing), q_m)),
                ManyOf(map(q_p_ -> Marginal(q_p_, false, false, nothing), q_p))
            )

            ref_val =
                0.6 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[1], q_p[1])), nothing)) +
                0.4 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[2], q_p[2])), nothing))
            @test score(AverageEnergy(), NormalMixture, Val{(:out, :switch, :m, :p)}(), marginals, nothing) ≈ ref_val
        end

        begin
            q_out = NormalMeanVariance(0.0, 1.0)
            q_switch = Categorical([0.5, 0.5])
            q_m = (NormalMeanPrecision(1.0, 2.0), NormalMeanPrecision(3.0, 4.0))
            q_p = (GammaShapeRate(3.0, 3.0), GammaShapeRate(4.0, 5.0))

            marginals = (
                Marginal(q_out, false, false, nothing),
                Marginal(q_switch, false, false, nothing),
                ManyOf(map(q_m_ -> Marginal(q_m_, false, false, nothing), q_m)),
                ManyOf(map(q_p_ -> Marginal(q_p_, false, false, nothing), q_p))
            )

            ref_val =
                0.5 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[1], q_p[1])), nothing)) +
                0.5 * (score(AverageEnergy(), NormalMeanPrecision, Val{(:out, :μ, :τ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[2], q_p[2])), nothing))
            @test score(AverageEnergy(), NormalMixture, Val{(:out, :switch, :m, :p)}(), marginals, nothing) ≈ ref_val
        end

        begin
            q_out = MvNormalMeanCovariance([0.0], [1.0])
            q_switch = Categorical([0.5, 0.5])
            q_m = (MvNormalMeanPrecision([1.0], [2.0]), MvNormalMeanPrecision([3.0], [4.0]))
            q_p = (WishartFast(3.0, fill(3.0, 1, 1)), WishartFast(4.0, fill(5.0, 1, 1)))

            marginals = (
                Marginal(q_out, false, false, nothing),
                Marginal(q_switch, false, false, nothing),
                ManyOf(map(q_m_ -> Marginal(q_m_, false, false, nothing), q_m)),
                ManyOf(map(q_p_ -> Marginal(q_p_, false, false, nothing), q_p))
            )

            ref_val =
                0.5 * (score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[1], q_p[1])), nothing)) +
                0.5 * (score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), map((q) -> Marginal(q, false, false, nothing), (q_out, q_m[2], q_p[2])), nothing))
            @test score(AverageEnergy(), NormalMixture, Val{(:out, :switch, :m, :p)}(), marginals, nothing) ≈ ref_val
        end
    end
end
