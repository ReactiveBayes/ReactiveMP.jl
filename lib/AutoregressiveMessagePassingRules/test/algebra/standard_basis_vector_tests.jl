# Every product is checked against the dense vector, whether a method of the package's or the
# generic fallback computes it.

@testitem "StandardBasisVector" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, LinearAlgebra, StableRNGs
    using AutoregressiveMessagePassingRules: StandardBasisVector

    rng = StableRNG(1234)

    @test_throws ArgumentError StandardBasisVector(0, 1)
    @test_throws ArgumentError StandardBasisVector(-10, 1)
    @test_throws ArgumentError StandardBasisVector(10, 11)
    @test_throws ArgumentError StandardBasisVector(10, -2)
    @test StandardBasisVector(3, 2) == [0, 1, 0]
    @test_throws BoundsError StandardBasisVector(3, 2)[4]

    for T in (Int, Float64, Float32)
        r = rand(rng, T)
        e = StandardBasisVector(2, 1, r)
        @test eltype(e) === T
        @test e[1] === r
        @test e[2] === zero(r)
    end

    # Same sizes
    for N in 1:8, I in 1:N, T in (Int, Float64, Float32)
        scale = rand(rng, T)
        e = StandardBasisVector(N, I, scale)
        e_c = zeros(T, N)
        e_c[I] = scale

        m = rand(rng, T)
        v = rand(rng, T, N)
        A = rand(rng, T, N, N)
        a = rand(rng, T, N, 1)

        @test m * e == m * e_c
        @test (m * e) isa StandardBasisVector
        @test m * e' == m * e_c'
        @test e * m == e_c * m
        @test (e * m) isa StandardBasisVector
        @test e' * m == e_c' * m

        for A in (A, Diagonal(diag(A)), A')
            @test (A * e) == (A * e_c)
            @test (A' * e) == (A' * e_c)
            @test (e * e') == (e_c * e_c')
            @test (e' * e) == (e_c' * e_c)
            @test (v' * e) == (v' * e_c)
            @test (e' * v) == (e_c' * v)
            @test (a' * e) == (a' * e_c)
            @test (a * e') == (a * e_c')
            @test (v * e') == (v * e_c')

            @test dot(e, A, e) === dot(e_c, A, e_c)
            @test dot(e, e) === dot(e_c, e_c)
            @test dot(e, e_c) === dot(e_c, e_c)
            @test dot(e_c, e) === dot(e_c, e_c)
            @test dot(v, e) === dot(v, e_c)
            @test dot(e, v) === dot(e_c, v)
            @test dot(v, e') === dot(v, e_c')
            @test dot(e', v) === dot(e_c', v)
            @test dot(v', e) === dot(v', e_c)
            @test dot(e, v') === dot(e_c, v')
            @test dot(v', e') === dot(v', e_c')
            @test dot(e', v') === dot(e_c', v')
        end
    end

    # Different sizes
    for N1 in 1:4, N2 in 1:4
        N1 == N2 && continue
        for I1 in 1:N1, I2 in 1:N2, T in (Int, Float64, Float32)
            scale1 = rand(rng, T)
            scale2 = rand(rng, T)
            e1 = StandardBasisVector(N1, I1, scale1)
            e2 = StandardBasisVector(N2, I2, scale2)
            e_c1 = zeros(T, N1)
            e_c1[I1] = scale1
            e_c2 = zeros(T, N2)
            e_c2[I2] = scale2

            @test_throws DimensionMismatch dot(e1, e2)
            @test_throws DimensionMismatch dot(e_c1, e2)
            @test_throws DimensionMismatch dot(e1, e_c2)
            @test_throws DimensionMismatch dot(e2, e1)
            @test_throws DimensionMismatch dot(e_c2, e1)
            @test_throws DimensionMismatch dot(e2, e_c1)
            @test_throws DimensionMismatch dot(e1, rand(rng, T, N1, N2 + 1), e2)
            @test_throws DimensionMismatch rand(rng, T, N2, N1 + 1) * e1

            @test e1 * e2' == e_c1 * e_c2'
            @test e2 * e1' == e_c2 * e_c1'
            @test e_c1 * e2' == e_c1 * e_c2'
            @test e_c2 * e1' == e_c2 * e_c1'
            @test e1 * e_c2' == e_c1 * e_c2'
            @test e2 * e_c1' == e_c2 * e_c1'
            @test dot(e1, rand(rng, T, N1, N2), e2) isa T
            @test_throws DimensionMismatch e1' * e2
            @test_throws DimensionMismatch e1' * e_c2
            @test_throws DimensionMismatch e2' * e1
            @test_throws DimensionMismatch e2' * e_c1
        end
    end

    # The promoted element type of the products the AR rules form.
    e = StandardBasisVector(3, 2, 1.0f0)
    @test eltype(rand(rng, 2, 3) * e) === Float64
    @test eltype(rand(rng, 3) * e') === Float64
    @test eltype(rand(rng, BigFloat, 2, 3) * e) === BigFloat
end

@testitem "StandardBasisVector:v_a_vT" tags = [:rules] begin
    using AutoregressiveMessagePassingRules, StandardMessagePassingRules, LinearAlgebra
    using AutoregressiveMessagePassingRules: StandardBasisVector

    # e a eᵀ, `dot`'s precision: a diagonal with one entry.
    W = StandardMessagePassingRules.v_a_vT(StandardBasisVector(3, 2, 2.0), 0.5)
    @test W isa Diagonal && W == Diagonal([0.0, 2.0, 0.0])
end
