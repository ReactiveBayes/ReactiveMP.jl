@testitem "rules:Bilinear:in" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Structured VMP: (m_out::UnivariateNormalDistributionsFamily, q_a::PointMass)" begin
        # m(in) ∝ ∫ exp(a⋅out⋅in) m_out(out) d(out) = exp(a⋅μ_out⋅in + a²⋅v_out⋅in²/2)
        #       ⇒ NormalWeightedMeanPrecision(a⋅μ_out, -a²⋅v_out)
        # This is the GaBP message of arXiv:0810.1119 with a = -A[i, j]:
        #   ξ_ij = -A_ij⋅μ_{i∖j},  P_ij = -A_ij²/P_{i∖j}.
        @test_rules [check_type_promotion = true] Bilinear(:in, Marginalisation) [
            (
                input  = (m_out = NormalMeanVariance(0.0, 1.0), q_a = PointMass(1.0)),
                output = NormalWeightedMeanPrecision(0.0, -1.0),
            ),
            (
                input  = (m_out = NormalMeanVariance(2.0, 3.0), q_a = PointMass(1.0)),
                output = NormalWeightedMeanPrecision(2.0, -3.0),
            ),
            # Negative coupling — this is the actual usage, since a = -A[i, j].
            (
                input  = (m_out = NormalMeanVariance(2.0, 3.0), q_a = PointMass(-0.5)),
                output = NormalWeightedMeanPrecision(-1.0, -0.75),
            ),
            (
                input  = (m_out = NormalMeanPrecision(-1.0, 4.0), q_a = PointMass(-2.0)),
                output = NormalWeightedMeanPrecision(2.0, -1.0),
            ),
            (
                input  = (m_out = NormalWeightedMeanPrecision(1.0, 2.0), q_a = PointMass(3.0)),
                output = NormalWeightedMeanPrecision(1.5, -4.5),
            ),
            # A zero coupling decouples the two variables entirely.
            (
                input  = (m_out = NormalMeanVariance(2.0, 3.0), q_a = PointMass(0.0)),
                output = NormalWeightedMeanPrecision(0.0, 0.0),
            ),
        ]
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
