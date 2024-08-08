
@testitem "Binomial Node" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, StableRNGs

    @testset "AverageEnergy with PointMasses" begin
        for k in (1, 3), n in (5, 10), p in (0.1, 0.8)
            q_k = PointMass(k)
            q_n = PointMass(n)
            q_p = PointMass(p)

            marginals = (Marginal(q_n, false, false, nothing), Marginal(q_k, false, false, nothing), Marginal(q_p, false, false, nothing))

            @test score(AverageEnergy(), Binomial, Val{(:n, :k, :p)}(), marginals, nothing) ≈ -logpdf(Binomial(n, p), k) atol = 1e-9
        end
    end

    @testset "AverageEnergy with PointMasses and Beta" begin
        rng = StableRNG(42)
        qks = (PointMass(5), PointMass(2), PointMass(7))
        qns = (PointMass(10), PointMass(12), PointMass(18))
        qps = (Beta(2, 3), Beta(1, 5), Beta(5, 1))
        for q_k in qks, q_n in qns, q_p in qps
            psamples = rand(rng, q_p, 30000)
            mc_estimate = mean(map((p) -> -logpdf(Binomial(mean(q_n), p), mean(q_k)), psamples))
            marginals = (Marginal(q_n, false, false, nothing), Marginal(q_k, false, false, nothing), Marginal(q_p, false, false, nothing))

            @test score(AverageEnergy(), Binomial, Val{(:n, :k, :p)}(), marginals, nothing) ≈ mc_estimate atol = 1e-1
        end
    end
end
