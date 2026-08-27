@testitem "rules:Bilinear:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Structured VMP: (m_in::UnivariateNormalDistributionsFamily, q_a::PointMass)" begin
        # m(out) ∝ ∫ exp(a⋅out⋅in) m_in(in) d(in) = exp(a⋅μ_in⋅out + a²⋅v_in⋅out²/2)
        #        ⇒ NormalWeightedMeanPrecision(a⋅μ_in, -a²⋅v_in)
        # This is the GaBP message of arXiv:0810.1119 with a = -A[i, j]:
        #   ξ_ij = -A_ij⋅μ_{i∖j},  P_ij = -A_ij²/P_{i∖j}.
        @test_rules [check_type_promotion = true] Bilinear(
            :out, Marginalisation
        ) [
            (
                input  = (m_in = NormalMeanVariance(0.0, 1.0), q_a = PointMass(1.0)),
                output = NormalWeightedMeanPrecision(0.0, -1.0),
            ),
            (
                input  = (m_in = NormalMeanVariance(2.0, 3.0), q_a = PointMass(1.0)),
                output = NormalWeightedMeanPrecision(2.0, -3.0),
            ),
            # Negative coupling — this is the actual usage, since a = -A[i, j].
            (
                input  = (m_in = NormalMeanVariance(2.0, 3.0), q_a = PointMass(-0.5)),
                output = NormalWeightedMeanPrecision(-1.0, -0.75),
            ),
            (
                input  = (m_in = NormalMeanPrecision(-1.0, 4.0), q_a = PointMass(-2.0)),
                output = NormalWeightedMeanPrecision(2.0, -1.0),
            ),
            (
                input  = (m_in = NormalWeightedMeanPrecision(1.0, 2.0), q_a = PointMass(3.0)),
                output = NormalWeightedMeanPrecision(1.5, -4.5),
            ),
            # A zero coupling decouples the two variables entirely.
            (
                input  = (m_in = NormalMeanVariance(2.0, 3.0), q_a = PointMass(0.0)),
                output = NormalWeightedMeanPrecision(0.0, 0.0),
            ),
        ]
    end

    @testset "Symmetry in `out` and `in`" begin
        # φ(out, in, a) = exp(out⋅a⋅in) is symmetric, so both message rules must agree.
        for (m, a) in (
            (NormalMeanVariance(1.5, 2.0), 0.75),
            (NormalMeanPrecision(-2.0, 3.0), -1.25),
        )
            @test (@call_rule Bilinear(:out, Marginalisation) (
                m_in = m, q_a = PointMass(a)
            )) ≈ (@call_rule Bilinear(:in, Marginalisation) (
                m_out = m, q_a = PointMass(a)
            ))
        end
    end

    @testset "Cross-check against NormalMeanPrecision" begin
        # The Gaussian density factorizes as
        #   N(out; in, w⁻¹) ∝ exp(-w⋅out²/2) ⋅ exp(w⋅out⋅in) ⋅ exp(-w⋅in²/2),
        # so the Bilinear factor with a = w is exactly the Gaussian cross-term. Tilting
        # the incoming message precision by +w and multiplying the Bilinear message by
        # NormalWeightedMeanPrecision(0, w) must reproduce the NormalMeanPrecision
        # message towards `out`: N(out; mean(m_μ), var(m_μ) + 1/w).
        for (ξ, w_in, w) in ((0.5, 2.0, 1.0), (-1.0, 4.0, 0.5))
            m_in        = NormalWeightedMeanPrecision(ξ, w_in)
            m_in_tilted = NormalWeightedMeanPrecision(ξ, w_in + w)
            expected    = @call_rule NormalMeanPrecision(:out, Marginalisation) (m_μ = m_in, m_τ = PointMass(w))
            @test all(
                mean_var(
                    prod(
                        GenericProd(),
                        NormalWeightedMeanPrecision(0.0, w),
                        @call_rule(
                            Bilinear(:out, Marginalisation),
                            (m_in = m_in_tilted, q_a = PointMass(w))
                        )
                    ),
                ) .≈ mean_var(expected),
            )
        end
    end
end
