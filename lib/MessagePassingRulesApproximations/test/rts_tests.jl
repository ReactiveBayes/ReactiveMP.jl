@testitem "smoothRTS: no cross-covariance, or a non-finite forward covariance, leaves the forward marginal" tags = [:approximations] begin
    using MessagePassingRulesApproximations, LinearAlgebra

    import MessagePassingRulesApproximations: smoothRTS

    # With no cross-covariance between `in` and `g(in)`, as a zero-covariance input gives, the
    # backward message on `out` says nothing about `in`: the smoothed marginal is the forward one,
    # with no NaN from inverting a zero `V_tilde`. A non-finite `V_tilde` leaves it too.

    @testset "scalar, zero V_tilde" begin
        m_in, V_in = smoothRTS(4.0, 0.0, 0.0, 2.0, 3.0, 5.0, 1.0)
        @test m_in == 2.0    # the forward mean
        @test V_in == 3.0    # the forward covariance
        @test !isnan(m_in)
        @test !isnan(V_in)
    end

    @testset "scalar, non-finite V_tilde" begin
        for bad in (Inf, NaN)
            m_in, V_in = smoothRTS(4.0, bad, 0.0, 2.0, 3.0, 5.0, 1.0)
            @test m_in == 2.0
            @test V_in == 3.0
        end
    end

    @testset "matrix, singular V_tilde" begin
        m_fw = [1.0, -2.0]
        V_fw = [2.0 0.5; 0.5 1.0]

        # Exactly zero
        m_in, V_in = smoothRTS(
            [0.0, 0.0],
            zeros(2, 2),
            zeros(2, 2),
            m_fw,
            V_fw,
            [1.0, 1.0],
            [1.0 0.0; 0.0 1.0],
        )
        @test m_in == m_fw
        @test V_in == V_fw

        # Rank-deficient, with no cross-covariance
        singular = [1.0 1.0; 1.0 1.0]
        m_in, V_in = smoothRTS(
            [0.0, 0.0],
            singular,
            zeros(2, 2),
            m_fw,
            V_fw,
            [1.0, 1.0],
            [1.0 0.0; 0.0 1.0],
        )
        @test m_in == m_fw
        @test V_in == V_fw
    end

    @testset "an invertible V_tilde gives the RTS equations' result" begin
        # Reference values from the RTS equations with the gain C V_tilde⁻¹, which an invertible
        # V_tilde allows.
        m_tilde, V_tilde, C_tilde = 4.0, 2.0, 1.5
        m_fw_in, V_fw_in = 2.0, 3.0
        m_bw_out, V_bw_out = 5.0, 1.0

        P = inv(V_tilde + V_bw_out)
        W_tilde = inv(V_tilde)
        D_tilde = C_tilde * W_tilde
        expected_V = V_fw_in + D_tilde * (V_bw_out * P * C_tilde - C_tilde)
        m_out = V_tilde * P * m_bw_out + V_bw_out * P * m_tilde
        expected_m = m_fw_in + D_tilde * (m_out - m_tilde)

        m_in, V_in = smoothRTS(
            m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out
        )

        @test m_in ≈ expected_m
        @test V_in ≈ expected_V
        # And it genuinely moved off the forward statistics, so the test above is not vacuous.
        @test m_in != m_fw_in
    end
end

@testitem "smoothRTS: a singular forward covariance, from a linearisation into more outputs" tags = [:approximations] begin
    using MessagePassingRulesApproximations, LinearAlgebra

    import MessagePassingRulesApproximations: smoothRTS

    # out = A in + b with three outputs from two inputs: the linearisation's V_tilde = A V Aᵀ has
    # rank 2. The smoothed marginal is the exact posterior of `in` given a normal backward message
    # on `out`, its precision V⁻¹ + Aᵀ V_bw⁻¹ A, and it is symmetric.
    A, b = [1.0 2.0; -1.0 0.5; 0.3 -1.2], [0.5, -1.0, 2.0]
    m_fw, V_fw = [1.0, -2.0], [2.0 0.5; 0.5 1.0]
    m_bw, V_bw = [3.0, 1.0, -0.5], [1.0 0.2 0.0; 0.2 0.5 0.1; 0.0 0.1 2.0]
    m_tilde, V_tilde, C_tilde = A * m_fw + b, A * V_fw * A', V_fw * A'
    @test rank(V_tilde) == 2

    m_in, V_in = smoothRTS(m_tilde, V_tilde, C_tilde, m_fw, V_fw, m_bw, V_bw)
    Λ = inv(V_fw) + A' * inv(V_bw) * A
    @test V_in ≈ inv(Λ)
    @test m_in ≈ Λ \ (V_fw \ m_fw + A' * (V_bw \ (m_bw - b)))
    @test norm(V_in - V_in') <= 1.0e-12 * norm(V_in)

    # The same through the package's linearisation of a map into three distances.
    beacons = ([0.0, 0.0], [10.0, 0.0], [0.0, 10.0])
    g(z) = [norm(z - s) for s in beacons]
    J, c = approximate(Linearization(), g, (m_fw,))
    m_tilde, V_tilde, C_tilde = J * m_fw + c, J * V_fw * J', V_fw * J'
    @test rank(V_tilde) == 2
    m_in, V_in = smoothRTS(m_tilde, V_tilde, C_tilde, m_fw, V_fw, m_bw, V_bw)
    @test norm(V_in - V_in') <= 1.0e-12 * norm(V_in)
    @test isposdef(Symmetric(V_in)) && all(<(0), eigvals(Symmetric(V_in - V_fw)) .- 1.0e-12)
end

@testitem "Unscented: a zero-covariance input gives a zero cross-covariance, not `nothing`" tags = [:approximations] begin
    using MessagePassingRulesApproximations, LinearAlgebra, Logging

    import MessagePassingRulesApproximations: Unscented, unscented_statistics

    # `__unscented_parameters_zero_covariance` returns a genuine zero cross-covariance, not
    # `nothing` ("not computed"): callers that request it (`Val(true)`) do arithmetic with it.

    @testset "univariate" begin
        (m, V, C) = with_logger(SimpleLogger(IOBuffer())) do
            unscented_statistics(
                Unscented(), Val(true), (x) -> x^2, (1.0,), (0.0,)
            )
        end

        @test m == 1.0            # g(1) = 1
        @test iszero(V)
        @test C !== nothing
        @test iszero(C)
        # Type-stable with the non-degenerate path, which returns a `Float64`.
        @test C isa Real
    end

    @testset "multivariate" begin
        (m, V, C) = with_logger(SimpleLogger(IOBuffer())) do
            unscented_statistics(
                Unscented(),
                Val(true),
                (x) -> x .^ 2,
                ([1.0, 2.0],),
                (zeros(2, 2),),
            )
        end

        @test m == [1.0, 4.0]
        @test all(iszero, V)
        @test C !== nothing
        @test all(iszero, C)
    end

    @testset "the non-degenerate path still returns a real cross-covariance" begin
        (m, V, C) = unscented_statistics(
            Unscented(), Val(true), (x) -> x^2, (1.0,), (2.0,)
        )
        @test C isa Real
        @test !iszero(C)
        @test isfinite(C)
    end
end
