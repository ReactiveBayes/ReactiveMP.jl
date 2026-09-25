@testitem "rules:AR:marginals" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, LinearAlgebra, StableRNGs

    @testset "y_x: (m_y, m_x, q_θ, q_γ), univariate" begin
        algorithm = ARVMP(Univariate, 1, ARsafe())
        @test_marginal_update_rule(
            node = AR, target = (:y, :x), algorithm = algorithm,
            cases = [
                (m = (y = NormalMeanPrecision(0.0, 1.0), x = NormalMeanPrecision(0.0, 1.0)), q = (θ = NormalMeanPrecision(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0))) =>
                    MvNormalWeightedMeanPrecision(zeros(2), [2.0 -1.0; -1.0 3.0]),
                (m = (y = NormalMeanPrecision(0.0, 1.0), x = NormalMeanPrecision(0.0, 1.0)), q = (θ = GammaShapeRate(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0))) =>
                    MvNormalWeightedMeanPrecision(zeros(2), [2.0 -1.0; -1.0 3.0]),
                (m = (y = NormalMeanPrecision(0.0, 1.0), x = NormalMeanPrecision(0.0, 1.0)), q = (θ = GammaShapeRate(1.0, 1.0), γ = NormalMeanPrecision(1.0, 1.0))) =>
                    MvNormalWeightedMeanPrecision(zeros(2), [2.0 -1.0; -1.0 3.0]),
            ],
        )
        # The multivariate joint has no table: the regularising precision on the noiseless
        # components makes its values unstable.
    end

    # For a univariate AR(1), where ARsafe regularises nothing, ARunsafe's joint by the Kalman
    # gain (see `ar_joint` in src/autoregressive.jl) must equal ARsafe's: for the first case its
    # covariance is the inverse of the precision [2 -1; -1 3], [0.6 0.2; 0.2 0.4].
    @testset "y_x: ARunsafe agrees with ARsafe, univariate" begin
        for (m_y, m_x, q_θ, q_γ) in (
                (NormalMeanPrecision(0.0, 1.0), NormalMeanPrecision(0.0, 1.0), NormalMeanPrecision(1.0, 1.0), GammaShapeRate(1.0, 1.0)),
                (NormalMeanVariance(0.3, 2.0), NormalMeanPrecision(-1.2, 0.7), NormalMeanVariance(0.6, 0.2), GammaShapeRate(3.0, 2.0)),
            )
            safe = call_marginal_update_rule(AR, (:y, :x); m = (y = m_y, x = m_x), q = (θ = q_θ, γ = q_γ), algorithm = ARVMP(Univariate, 1, ARsafe()))
            unsafe = call_marginal_update_rule(AR, (:y, :x); m = (y = m_y, x = m_x), q = (θ = q_θ, γ = q_γ), algorithm = ARVMP(Univariate, 1, ARunsafe()))
            @test unsafe isa MvNormalMeanCovariance
            @test mean(unsafe) ≈ mean(safe)
            @test cov(unsafe) ≈ cov(safe)
        end
        @test_marginal_update_rule(
            node = AR, target = (:y, :x), algorithm = ARVMP(Univariate, 1, ARunsafe()),
            cases = [
                (m = (y = NormalMeanPrecision(0.0, 1.0), x = NormalMeanPrecision(0.0, 1.0)), q = (θ = NormalMeanPrecision(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0))) =>
                    MvNormalMeanCovariance(zeros(2), [0.6 0.2; 0.2 0.4]),
            ],
        )
    end

    # For a multivariate AR, y[2:end] is x[1:end-1] exactly, so the joint is that of z = (y₁, x),
    # of dimension order + 1, mapped by (y, x) = L z. Its precision and weighted mean are
    #
    #     Tᵀ Wy T + [0 0; 0 Wx + ⟨γ⟩Vθ] + ⟨γ⟩ (1, -θ)(1, -θ)ᵀ,   Tᵀ Wy my + (0, Wx mx),
    #
    # with T z = y, an exact reference ARunsafe must match. ARsafe gives the noiseless components
    # the finite precision `huge` instead, and matches it only to about 1e-3 relative (measured).
    function exact_joint(order, m_y, m_x, q_θ, q_γ)
        (my, Wy), (mx, Wx) = mean_invcov(m_y), mean_invcov(m_x)
        (mθ, Vθ), mγ = mean_cov(q_θ), mean(q_γ)
        T = zeros(order, order + 1)
        T[1, 1] = 1
        for i in 2:order
            T[i, i] = 1
        end
        L = [T; zeros(order, 1) Matrix(1.0I, order, order)]
        u = [1; -mθ]
        W = T' * Wy * T + mγ * u * u'
        W[2:end, 2:end] += Wx + mγ * Vθ
        ξ = T' * Wy * my
        ξ[2:end] += Wx * mx
        Σz = inv(W)
        return L * (Σz * ξ), L * Σz * L'
    end

    @testset "y_x: ARunsafe is exact, ARsafe close to it, multivariate" begin
        rng = StableRNG(42)
        for order in (2, 3)
            B = randn(rng, order, order)
            m_y = MvNormalMeanCovariance(randn(rng, order), B * B' + I)
            B = randn(rng, order, order)
            m_x = MvNormalMeanCovariance(randn(rng, order), B * B' + I)
            q_θ = MvNormalMeanCovariance(randn(rng, order), Matrix(0.1I, order, order))
            q_γ = GammaShapeRate(3.0, 2.0)
            μ, Σ = exact_joint(order, m_y, m_x, q_θ, q_γ)
            safe = call_marginal_update_rule(AR, (:y, :x); m = (y = m_y, x = m_x), q = (θ = q_θ, γ = q_γ), algorithm = ARVMP(Multivariate, order, ARsafe()))
            unsafe = call_marginal_update_rule(AR, (:y, :x); m = (y = m_y, x = m_x), q = (θ = q_θ, γ = q_γ), algorithm = ARVMP(Multivariate, order, ARunsafe()))
            @test unsafe isa MvNormalMeanCovariance
            @test isapprox(mean(unsafe), μ; atol = 1.0e-10)
            @test isapprox(cov(unsafe), Σ; atol = 1.0e-10)
            @test isapprox(mean(safe), μ; rtol = 1.0e-3)
            @test isapprox(cov(safe), Σ; rtol = 1.0e-3)
        end
    end
end
