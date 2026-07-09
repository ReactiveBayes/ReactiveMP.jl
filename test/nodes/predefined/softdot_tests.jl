
@testitem "SoftDotNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, LinearAlgebra

    # Closed-form mean-field average energy for `softdot`, written out explicitly from
    #     U = ½( log2π − ⟨log γ⟩ + ⟨γ⟩·⟨(y − θᵀx)²⟩ ),
    # where the whole ⟨(y − θᵀx)²⟩ bracket is multiplied by a *single* power of ⟨γ⟩.
    # Used as an independent reference for the mean-field @average_energy rule. This
    # anchors the regression for the historical bug in which the cross term carried a
    # spurious extra ⟨γ⟩ (see issue #615).
    function softdot_meanfield_ae_reference(q_y, q_θ, q_x, q_γ)
        m_y, V_y = mean_cov(q_y)
        m_θ, V_θ = mean_cov(q_θ)
        m_x, V_x = mean_cov(q_x)
        Eyθx =
            V_y + m_y^2 - 2 * m_y * dot(m_θ, m_x) +
            tr(V_θ * V_x) +
            dot(m_x, V_θ, m_x) +
            dot(m_θ, V_x + m_x * m_x', m_θ)
        return (log(2π) - mean(log, q_γ) + mean(q_γ) * Eyθx) / 2
    end

    softdot_meanfield_ae(q_y, q_θ, q_x, q_γ) = score(
        AverageEnergy(),
        SoftDot,
        Val{(:y, :θ, :x, :γ)}(),
        (Marginal(q_y, false, false), Marginal(q_θ, false, false), Marginal(q_x, false, false), Marginal(q_γ, false, false)),
        nothing,
    )

    softdot_structured_ae(q_y_x, q_θ, q_γ) = score(
        AverageEnergy(),
        SoftDot,
        Val{(:y_x, :θ, :γ)}(),
        (Marginal(q_y_x, false, false), Marginal(q_θ, false, false), Marginal(q_γ, false, false)),
        nothing,
    )

    @testset "AverageEnergy: mean-field variant matches the closed-form reference" begin
        # Cover a range of `mean(q_γ)` values, including `mean(q_γ) ≠ 1`, which is where the
        # historical extra-`mean(q_γ)` bug on the cross term manifested.
        configs = (
            (NormalMeanVariance(3.0, 7.0), NormalMeanVariance(11.0, 13.0), NormalMeanVariance(5.0, 9.0), GammaShapeRate(3 / 2, 4242 / 2)),
            (NormalMeanVariance(0.4, 0.6), NormalMeanVariance(1.2, 0.25), NormalMeanVariance(0.5, 0.3), GammaShapeRate(4.0, 1.0)),
            (
                NormalMeanVariance(3.0, 7.0),
                MvNormalMeanCovariance([23.0, 29.0], [31.0 37.0; 37.0 43.0]),
                MvNormalMeanCovariance([5.0, 9.0], [11.0 13.0; 13.0 19.0]),
                GammaShapeRate(5.0, 2.0),
            ),
        )
        for (q_y, q_θ, q_x, q_γ) in configs
            @test softdot_meanfield_ae(q_y, q_θ, q_x, q_γ) ≈ softdot_meanfield_ae_reference(q_y, q_θ, q_x, q_γ)
        end
    end

    @testset "AverageEnergy: mean-field regression against a hand-computed value (issue #615)" begin
        # Exact reproducer from issue #615. The buggy implementation (extra `mean(q_γ)` on
        # the cross term) returned a value ½·⟨γ⟩·(⟨γ⟩−1)·(−2·m_y·m_θᵀm_x) = −2.88 too low here.
        q_y = NormalMeanVariance(0.4, 0.6)
        q_θ = NormalMeanVariance(1.2, 0.25)
        q_x = NormalMeanVariance(0.5, 0.3)
        q_γ = GammaShapeRate(4.0, 1.0) # mean(q_γ) = 4

        S = 0.6 + 0.4^2 - 2 * 0.4 * 1.2 * 0.5 + 0.25 * 0.3 + 0.5^2 * 0.25 + 1.2^2 * (0.3 + 0.5^2)
        ref = 0.5 * (log(2π) - mean(log, q_γ) + 4.0 * S)

        @test softdot_meanfield_ae(q_y, q_θ, q_x, q_γ) ≈ ref
    end

    @testset "AverageEnergy: mean-field and structured variants agree (zero y-x covariance)" begin
        # With no posterior covariance between y and x, the structured q(y, x) variant reduces
        # to the fully factorized mean-field one, so the two @average_energy methods must agree
        # for any `mean(q_γ)`. Under the issue #615 bug they disagreed whenever `mean(q_γ) ≠ 1`.
        q_γ = GammaShapeRate(4.0, 1.0) # mean(q_γ) = 4

        # Univariate x
        m_y, V_y = 0.4, 0.6
        m_x, V_x = 0.5, 0.3
        q_y = NormalMeanVariance(m_y, V_y)
        q_x = NormalMeanVariance(m_x, V_x)
        q_θ = NormalMeanVariance(1.2, 0.25)
        q_yx = MvNormalMeanCovariance([m_y, m_x], [V_y 0.0; 0.0 V_x])
        @test softdot_meanfield_ae(q_y, q_θ, q_x, q_γ) ≈ softdot_structured_ae(q_yx, q_θ, q_γ)

        # Multivariate x
        m_y2, V_y2 = 3.0, 7.0
        m_x2 = [5.0, 9.0]
        V_x2 = [11.0 4.0; 4.0 19.0]
        q_y2 = NormalMeanVariance(m_y2, V_y2)
        q_x2 = MvNormalMeanCovariance(m_x2, V_x2)
        q_θ2 = MvNormalMeanCovariance([1.2, 0.7], [0.25 0.05; 0.05 0.3])
        q_yx2 = MvNormalMeanCovariance([m_y2; m_x2], [V_y2 0.0 0.0; 0.0 V_x2[1, 1] V_x2[1, 2]; 0.0 V_x2[2, 1] V_x2[2, 2]])
        @test softdot_meanfield_ae(q_y2, q_θ2, q_x2, q_γ) ≈ softdot_structured_ae(q_yx2, q_θ2, q_γ)
    end

    @testset "AverageEnergy: structured variant" begin
        # The structured q(y, x) variant is unaffected by the issue #615 fix; these pin its values.
        begin
            q_y_x = MvNormalMeanCovariance(zeros(2), diageye(2))
            q_θ = NormalMeanVariance(0.0, 1.0)
            q_γ = GammaShapeRate(2.0, 3.0)
            @test softdot_structured_ae(q_y_x, q_θ, q_γ) ≈ 1.92351917665616
        end

        begin
            q_y_x = MvNormalMeanCovariance(zeros(3), diageye(3))
            q_θ = MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0])
            q_γ = GammaShapeRate(2.0, 3.0)
            @test softdot_structured_ae(q_y_x, q_θ, q_γ) ≈ 2.256852 atol = 1e-4
        end
    end # testset: structured
end # testset
