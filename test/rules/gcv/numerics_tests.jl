
@testitem "rules:GCV: extreme exponents do not produce NaN" setup = [
    GCVRulesTestUtils
] begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    import ReactiveMP: __gcv_log_noise_precision, GaussHermiteCubature

    default_meta = GCVRulesTestUtils.default_meta

    # The `GCV` node's effective noise precision is `⟨e^{-(κz+ω)}⟩ = A · B` with
    #
    #     A = exp(-⟨ω⟩ + Var(ω)/2)        B = exp(-⟨κ⟩⟨z⟩ + Var(κz)/2)
    #
    # Forming that product directly is unsafe in a way that has nothing to do with the size of
    # the answer: either factor can overflow or underflow *on its own* while `A · B` is perfectly
    # representable. The sharp case is `log A = +800`, `log B = -800`, where
    #
    #     A = Inf,  B = 0,  A * B = NaN
    #
    # but the true value is `exp(0) = 1`. Summing the exponents first gives the right answer.
    #
    # Issue #624 reported this class of failure. Its worked example (`ω_mean ≈ 40`) does not
    # actually overflow -- `A = exp(-40) ≈ 4e-18` and `inv(4e-18) ≈ 2.4e17`, both finite -- and
    # `:y` only reaches `Inf` around `ω_mean ≈ 740`, where the true variance genuinely exceeds
    # `floatmax` and `Inf` is the correct answer. The reachable defect is the `NaN` below.
    meta = default_meta()

    # log A = -(-800) = +800  ⇒  A = Inf
    # Var(κz) = 0, so log B = -1·800 = -800  ⇒  B = 0
    # log A + log B = 0  ⇒  precision = 1  ⇒  added variance = 1
    q_ω_big = NormalMeanVariance(-800.0, 0.0)
    q_z_big = NormalMeanVariance(800.0, 0.0)
    q_κ_one = NormalMeanVariance(1.0, 0.0)

    @testset "the helper cancels the exponents exactly" begin
        @test __gcv_log_noise_precision(q_z_big, q_κ_one, q_ω_big) == 0.0
        # And the naive product really is NaN, so this is not a hypothetical.
        A = exp(-mean(q_ω_big) + var(q_ω_big) / 2)
        B = exp(-mean(q_κ_one) * mean(q_z_big))
        @test isinf(A)
        @test iszero(B)
        @test isnan(A * B)
    end

    @testset ":y and :x messages stay finite" begin
        # `@call_rule` is a macro, so each signature is written out rather than splatted.
        # Added variance is `exp(-0) = 1` in every case; the message-based variants additionally
        # carry the incoming variance of 1.
        incoming = NormalMeanVariance(0.0, 1.0)

        y_from_message = @call_rule GCV(:y, Marginalisation) (
            m_x = incoming,
            q_z = q_z_big,
            q_κ = q_κ_one,
            q_ω = q_ω_big,
            meta = meta,
        )
        y_from_marginal = @call_rule GCV(:y, Marginalisation) (
            q_x = incoming,
            q_z = q_z_big,
            q_κ = q_κ_one,
            q_ω = q_ω_big,
            meta = meta,
        )
        x_from_message = @call_rule GCV(:x, Marginalisation) (
            m_y = incoming,
            q_z = q_z_big,
            q_κ = q_κ_one,
            q_ω = q_ω_big,
            meta = meta,
        )
        x_from_marginal = @call_rule GCV(:x, Marginalisation) (
            q_y = incoming,
            q_z = q_z_big,
            q_κ = q_κ_one,
            q_ω = q_ω_big,
            meta = meta,
        )

        for msg in (y_from_message, x_from_message)
            @test !isnan(var(msg))
            @test isfinite(var(msg))
            @test var(msg) ≈ 1.0 + 1.0
        end

        for msg in (y_from_marginal, x_from_marginal)
            @test !isnan(var(msg))
            @test isfinite(var(msg))
            @test var(msg) ≈ 1.0
        end
    end

    @testset "an asymmetric cancellation is also handled" begin
        # log A = +800, log B = -790  ⇒  sum = +10  ⇒  precision = e^10, variance = e^-10.
        # Naively: Inf * (something underflowed) = NaN.
        q_z = NormalMeanVariance(790.0, 0.0)
        msg = @call_rule GCV(:y, Marginalisation) (
            q_x = NormalMeanVariance(0.0, 1.0),
            q_z = q_z,
            q_κ = q_κ_one,
            q_ω = q_ω_big,
            meta = meta,
        )
        @test !isnan(var(msg))
        @test var(msg) ≈ exp(-10.0)
    end

    @testset "the y_x joint marginal stays finite and positive-definite" begin
        joint = @call_marginalrule GCV(:y_x) (
            m_y = NormalMeanVariance(1.0, 1.0),
            m_x = NormalMeanVariance(0.0, 1.0),
            q_z = q_z_big,
            q_κ = q_κ_one,
            q_ω = q_ω_big,
            meta = meta,
        )

        W = invcov(joint)
        @test all(!isnan, W)
        @test all(isfinite, W)
        @test isposdef(W)
        # Coupling is exactly `exp(0) = 1`
        @test W[1, 2] ≈ -1.0
        @test W[1, 1] ≈ 1.0 + 1.0
    end

    @testset "both average energies stay finite" begin
        q_y = NormalMeanVariance(1.0, 1.0)
        q_x = NormalMeanVariance(0.0, 1.0)
        psi = (1.0 - 0.0)^2 + 1.0 + 1.0

        marginals = map(
            q -> Marginal(q, false, false),
            (q_y, q_x, q_z_big, q_κ_one, q_ω_big),
        )
        ae_mf = score(
            AverageEnergy(), GCV, Val{(:y, :x, :z, :κ, :ω)}(), marginals, meta
        )

        # ⟨κz + ω⟩ = 1·800 + (-800) = 0, and psi·exp(0) = psi
        @test !isnan(ae_mf)
        @test isfinite(ae_mf)
        @test ae_mf ≈ (log(2π) + 0.0 + psi) / 2

        q_y_x = MvNormalMeanCovariance([1.0, 0.0], [1.0 0.0; 0.0 1.0])
        ae_st = score(
            AverageEnergy(),
            GCV,
            Val{(:y_x, :z, :κ, :ω)}(),
            map(
                q -> Marginal(q, false, false),
                (q_y_x, q_z_big, q_κ_one, q_ω_big),
            ),
            meta,
        )
        @test !isnan(ae_st)
        @test isfinite(ae_st)
        # Zero y-x covariance, so it must agree with the mean-field variant.
        @test ae_st ≈ ae_mf
    end

    @testset "a genuinely unrepresentable variance is still Inf, not NaN" begin
        # Here there is no cancellation to exploit: log A + log B = -740, so the true variance is
        # `exp(740)`, beyond `floatmax`. `Inf` is the honest answer -- log-space arithmetic
        # cannot conjure a representable result. What matters is that it is not `NaN`.
        msg = @call_rule GCV(:y, Marginalisation) (
            q_x = NormalMeanVariance(0.0, 1.0),
            q_z = NormalMeanVariance(0.0, 0.0),
            q_κ = NormalMeanVariance(0.0, 0.0),
            q_ω = NormalMeanVariance(740.0, 0.0),
            meta = meta,
        )
        @test !isnan(var(msg))
        @test isinf(var(msg))
        @test var(msg) > 0
    end
end
