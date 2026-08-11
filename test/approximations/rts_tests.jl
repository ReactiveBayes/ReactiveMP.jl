
@testitem "smoothRTS: a singular forward covariance yields the forward marginal unchanged" begin
    using ReactiveMP, BayesBase, Distributions, ExponentialFamily, LinearAlgebra

    import ReactiveMP: smoothRTS

    # If the transformed input has zero (or non-finite) covariance, the node's output carries no
    # uncertainty, so the backward message cannot revise the input: the smoothed inbound
    # marginal is exactly the forward one.
    #
    # Previously `W_tilde = cholinv(V_tilde)` produced `Inf` rather than raising -- `cholinv` on
    # a scalar is just `inv` -- so `D_tilde = C_tilde * W_tilde` became `0 * Inf = NaN` and a
    # silently corrupted marginal propagated.

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

        # Rank-deficient but non-zero
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

    @testset "a well-conditioned V_tilde is unaffected" begin
        # The guard must not alter the normal path. Reference values computed from the RTS
        # equations directly.
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

@testitem "Unscented: a zero-covariance input gives a zero cross-covariance, not `nothing`" begin
    using ReactiveMP,
        BayesBase, Distributions, ExponentialFamily, LinearAlgebra, Logging

    import ReactiveMP: Unscented, unscented_statistics

    # `__unscented_parameters_zero_covariance` returned `nothing` for the cross-covariance --
    # "not computed" rather than a degenerate value of the same kind as the zeros beside it.
    # Callers that legitimately requested it (`Val(true)`) then fed `nothing` into arithmetic
    # (issue #630).

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

@testitem "DeltaFn(:ins) marginal completes for a zero-variance inbound" begin
    using ReactiveMP,
        BayesBase, Distributions, ExponentialFamily, LinearAlgebra, Logging

    import ReactiveMP: DeltaMeta, Unscented, Linearization

    # The reachable path reported in #630: `@marginalrule DeltaFn(:ins)` feeds
    # `unscented_statistics`' third return value straight into `smoothRTS`, so a zero-variance
    # inbound message used to fail with
    #   MethodError: no method matching *(::Nothing, ::Float64)
    #
    # A zero-variance inbound means the input is known exactly, so the correct smoothed marginal
    # is that same point -- which is what both approximation methods now return.
    for method in (Unscented(), Linearization())
        @testset "$(nameof(typeof(method)))" begin
            meta = DeltaMeta(method = method, inverse = nothing)

            result = with_logger(SimpleLogger(IOBuffer())) do
                @call_marginalrule DeltaFn{(x) -> x^2}(:ins) (
                    m_out = NormalMeanVariance(2.0, 1.0),
                    m_ins = ManyOf(NormalMeanVariance(1.0, 0.0)),
                    meta = meta,
                )
            end

            m, V = mean_cov(result)
            @test m == 1.0        # unchanged: the input was already deterministic
            @test iszero(V)
            @test !isnan(m)
            @test !isnan(V)
        end
    end

    @testset "a non-degenerate inbound is unaffected" begin
        # Guards against the short-circuit swallowing the normal path.
        meta = DeltaMeta(method = Unscented(), inverse = nothing)
        result = @call_marginalrule DeltaFn{(x) -> x^2}(:ins) (
            m_out = NormalMeanVariance(2.0, 1.0),
            m_ins = ManyOf(NormalMeanVariance(1.0, 0.5)),
            meta = meta,
        )

        m, V = mean_cov(result)
        @test isfinite(m)
        @test V > 0
        # The backward message did move the marginal away from the forward input.
        @test m != 1.0
        @test V != 0.5
    end
end
