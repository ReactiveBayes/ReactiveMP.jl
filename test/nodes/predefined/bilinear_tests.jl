@testitem "BilinearNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily

    @testset "AverageEnergy" begin
        # U = -E[log φ] = -mean(q_a) * E_{q(out, in)}[out * in]
        #   = -mean(q_a) * (V[1, 2] + m[1] * m[2])  where  m, V = mean_cov(q_out_in)
        begin
            q_out_in = MvNormalMeanCovariance([1.0, 2.0], [2.0 0.5; 0.5 3.0])
            q_a      = PointMass(2.0)

            marginals = (
                Marginal(q_out_in, false, false), Marginal(q_a, false, false)
            )

            @test score(
                AverageEnergy(),
                Bilinear,
                Val{(:out_in, :a)}(),
                marginals,
                nothing,
            ) ≈ -5.0
        end
    end
end
