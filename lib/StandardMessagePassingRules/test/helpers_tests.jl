# The algebra helpers the node packages share, with a stable generator.

@testitem "helpers:algebra" tags = [:helpers] begin
    using StandardMessagePassingRules, LinearAlgebra, StableRNGs
    import StandardMessagePassingRules: negate_inplace!, mul_inplace!, rank1update, mul_trace

    rng = StableRNG(1234)
    for n in 2:4
        # An `Array` is overwritten; a number cannot be.
        A = rand(rng, n, n)
        B = copy(A)
        @test negate_inplace!(B) == -A && B == -A
        a = rand(rng)
        @test negate_inplace!(a) == -a

        B, c = copy(A), randn(rng)
        @test mul_inplace!(c, B) == c * A && B == c * A
        @test mul_inplace!(c, a) == c * a

        # Other arrays are left alone.
        view_of = view(copy(A), :, :)
        @test negate_inplace!(view_of) == -A
    end

    for n in 2:4, T1 in (Float32, Float64), T2 in (Float32, Float64), T3 in (Float32, Float64)
        A, x, y = rand(rng, T1, n, n), rand(rng, T2, n), rand(rng, T3, n)
        @test rank1update(A, x) ≈ A + x * x'
        @test rank1update(A, x, y) ≈ A + x * y'
        @test eltype(rank1update(A, x, y)) === promote_type(T1, T2, T3)
    end
    A = rand(rng, 3, 3)
    @test rank1update(A, [1.0, 2.0, 3.0]) !== A          # the BLAS path copies
    @test rank1update(2.0, 3.0) == 11.0 && rank1update(2.0, 3.0, 4.0) == 14.0

    for n in 2:4, T1 in (Float32, Float64), T2 in (Float32, Float64)
        A, B = rand(rng, T1, n, n), rand(rng, T2, n, n)
        @test mul_trace(A, B) ≈ tr(A * B)
        a, b = rand(rng, T1), rand(rng, T2)
        @test mul_trace(a, b) ≈ a * b
    end
end

@testitem "helpers:diageye" tags = [:helpers] begin
    using StandardMessagePassingRules

    # Exported, with its element type or, as models write it, without one.
    @test diageye(Float32, 2) == Float32[1 0; 0 1] && eltype(diageye(Float32, 2)) === Float32
    @test diageye(3) == [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] && eltype(diageye(3)) === Float64
end
