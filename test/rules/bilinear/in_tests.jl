@testitem "rules:Bilinear:in" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Structured VMP: (m_out::UnivariateNormalDistributionsFamily, q_a::PointMass)" begin
        @test (@call_rule Bilinear(:in, Marginalisation) (
            m_out = NormalMeanVariance(0.0, 1.0), q_a = PointMass(1.0)
        )) isa NormalWeightedMeanPrecision
    end

    @testset "Cross-check against NormalMeanPrecision" begin
        # The Gaussian density factorizes as
        #   N(out; in, w⁻¹) ∝ exp(-w⋅out²/2) ⋅ exp(w⋅out⋅in) ⋅ exp(-w⋅in²/2),
        # so the Bilinear factor with a = w is exactly the Gaussian cross-term. Tilting
        # the incoming message precision by +w and multiplying the Bilinear message by
        # NormalWeightedMeanPrecision(0, w) must reproduce the NormalMeanPrecision
        # message towards its mean interface: N(in; mean(m_out), var(m_out) + 1/w).
        for (ξ, w_out, w) in ((0.5, 2.0, 1.0), (-1.0, 4.0, 0.5))
            m_out        = NormalWeightedMeanPrecision(ξ, w_out)
            m_out_tilted = NormalWeightedMeanPrecision(ξ, w_out + w)
            expected     = @call_rule NormalMeanPrecision(:μ, Marginalisation) (m_out = m_out, m_τ = PointMass(w))
            @test all(
                mean_var(
                    prod(
                        GenericProd(),
                        NormalWeightedMeanPrecision(0.0, w),
                        @call_rule(
                            Bilinear(:in, Marginalisation),
                            (m_out = m_out_tilted, q_a = PointMass(w))
                        )
                    ),
                ) .≈ mean_var(expected),
            )
        end
    end
end
