# v6 had no tests of the companion matrix. Every product is checked against the dense matrix,
# whether a method of the port's or the generic fallback computes it.

@testitem "CompanionMatrix" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, LinearAlgebra, StableRNGs
    using AutoregressiveMessagePassingRules: CompanionMatrix, CompanionMatrixTransposed, as_companion_matrix

    function dense_companion(θ)
        n = length(θ)
        S = zeros(eltype(θ), n, n)
        S[1, :] = θ
        for i in 2:n
            S[i, i - 1] = one(eltype(θ))
        end
        return S
    end

    rng = StableRNG(42)

    @test as_companion_matrix(0.5) === 0.5
    @test as_companion_matrix([1.0, 2.0]) isa CompanionMatrix{Float64}

    for n in 1:5, T in (Float64, Float32)
        θ = randn(rng, T, n)
        A = as_companion_matrix(θ)
        S = dense_companion(θ)

        @test size(A) == (n, n)
        @test eltype(A) === T
        @test A == S
        @test Matrix(A) == S
        @test A' isa CompanionMatrixTransposed
        @test transpose(A) isa CompanionMatrixTransposed
        @test A' == S'
        @test (A')' isa CompanionMatrix
        @test transpose(transpose(A)) isa CompanionMatrix
        @test (A')' == S
        @test_throws BoundsError A[n + 1, 1]
        @test_throws BoundsError A'[1, n + 1]

        # θₙ ≠ 0 for these draws, so the companion matrix is invertible.
        @test inv(A) ≈ inv(S)
        @test inv(A') ≈ inv(Matrix(S'))

        for R in (Float64, Float32, BigFloat)
            v = randn(rng, R, n)
            M = randn(rng, R, n, n + 1)
            N = randn(rng, R, n + 1, n)
            P = promote_type(T, R)

            @test A * v ≈ S * v
            @test A * M ≈ S * M
            @test N * A ≈ N * S
            @test A' * M ≈ S' * M
            @test N * A' ≈ N * S'
            @test eltype(A * v) === P
            @test eltype(A * M) === P
            @test eltype(N * A) === P
            @test eltype(A' * M) === P
            @test eltype(N * A') === P
            # The products the AR rules chain.
            @test A * M[:, 1:n] * A' ≈ S * M[:, 1:n] * S'
            @test A' * M[:, 1:n] * A ≈ S' * M[:, 1:n] * S
        end

        # Other operands take the generic fallback.
        D = Diagonal(randn(rng, n))
        @test A * D ≈ S * D
        @test D * A' ≈ D * S'
        V = randn(rng, n + 1, n + 1)
        @test A * view(V, 1:n, 1:n) ≈ S * V[1:n, 1:n]
        @test A * view(V, 1:n, 1) ≈ S * V[1:n, 1]

        @test_throws DimensionMismatch A * randn(rng, n + 1)
        @test_throws DimensionMismatch A * randn(rng, n + 1, 2)
        @test_throws DimensionMismatch randn(rng, 2, n + 1) * A
        @test_throws DimensionMismatch A' * randn(rng, n + 1, 2)
        @test_throws DimensionMismatch randn(rng, 2, n + 1) * A'
    end
end
