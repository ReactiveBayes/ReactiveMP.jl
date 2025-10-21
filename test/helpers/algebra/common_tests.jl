
@testitem "AlgebraCommonTest" begin
    using ReactiveMP
    using Random
    using LinearAlgebra

    @testset "negate_inplace!" begin
        rng = MersenneTwister(1234)

        for size in 2:4
            A = rand(rng, size, size)
            B = copy(A)
            @test B == A
            @test ReactiveMP.negate_inplace!(B) == -A
            @test B == -A

            a = rand(rng)
            b = copy(a)
            @test ReactiveMP.negate_inplace!(b) == -a
            @test b == a # numbers stay unchanged
        end
    end

    @testset "negate_inplace!" begin
        rng = MersenneTwister(1234)

        for size in 2:4
            A = rand(rng, size, size)
            B = copy(A)
            c = randn(rng)
            @test B == A
            @test ReactiveMP.mul_inplace!(c, B) == c * A
            @test B == c * A

            a = rand(rng)
            b = copy(a)
            c = randn(rng)
            @test ReactiveMP.mul_inplace!(c, b) == c * a
            @test b == a # numbers stay unchanged
        end
    end

    @testset "rank1update" begin
        rng = MersenneTwister(1234)

        for size in 2:4, T1 in (Float32, Float64), T2 in (Float32, Float64), T3 in (Float32, Float64)
            A = rand(rng, T1, size, size)
            x = rand(rng, T2, size)
            @test ReactiveMP.rank1update(A, x) ≈ (A + x * x')
            y = rand(rng, T3, size)
            @test ReactiveMP.rank1update(A, x, y) ≈ (A + x * y')
        end
    end

    @testset "mul_trace" begin
        rng = MersenneTwister(1234)

        for size in 2:4, T1 in (Float32, Float64), T2 in (Float32, Float64)
            A = rand(rng, T1, size, size)
            B = rand(rng, T2, size, size)

            @test ReactiveMP.mul_trace(A, B) ≈ tr(A * B)

            a = rand(rng, T1)
            b = rand(rng, T2)
            @test ReactiveMP.mul_trace(a, b) ≈ a * b
        end
    end

    @testset "v_a_vT" begin
        import ReactiveMP: v_a_vT

        for n in (1, 2, 3), v1 in [rand(n) for _ in 1:5], a in rand(5)
            @test v_a_vT(v1, a) ≈ v1 * a * v1'
            @test v_a_vT(v1, a) ≈ v1 * v1' * a
        end

        for n in (1, 2, 3), v1 in [rand(n) for _ in 1:5], v2 in [rand(n) for _ in 1:5], a in rand(5)
            @test v_a_vT(v1, a, v2) ≈ v1 * a * v2'
            @test v_a_vT(v1, a, v2) ≈ v1 * v2' * a
        end

        for n in (1, 2, 3), v in [rand(1, n) for _ in 1:5], a in [rand(n, n) for _ in 1:5]
            @test v_a_vT(v, a) ≈ v * a * v'
        end

        for n in (1, 2, 3), v1 in [rand(1, n) for _ in 1:5], v2 in [rand(1, n) for _ in 1:5], a in [rand(n, n) for _ in 1:5]
            @test v_a_vT(v1, a, v2) ≈ v1 * a * v2'
        end
    end
end

@testitem "isonehot" begin
    import ReactiveMP: isonehot

    for T in [Float64, Float32, Float16, BigFloat]
        @test isonehot(T.([0.0, 1.0])) == true
        @test isonehot(T.([0.0, 1.0, 0.0])) == true
        @test isonehot(T.([1.0, 0.0, 0.0])) == true
        @test isonehot(T.([0.0, 0.0, 1.0])) == true

        @test isonehot(T.([0.0, 0.0, 0.0])) == false
        @test isonehot(T.([1.0, 1.0, 1.0])) == false

        @test isonehot(T.([0.0, 0.1, 1.0])) == false
        @test isonehot(T.([0.1, 0.1, 0.8])) == false

        if T !== BigFloat
            v = T.([0.0, 1.0])
            @test @allocated(isonehot(v)) == 0
        end
    end
end

@testitem "isonehot approx" begin
    import ReactiveMP: isonehot

    @test isonehot([2.9999999999849994e-12, 2.9999999999849994e-12, 0.999999999994]) == true
    @test isonehot([2.9999999999849994e-12, 0.999999999994, 2.9999999999849994e-12]) == true
    @test isonehot([0.999999999994, 2.9999999999849994e-12, 2.9999999999849994e-12]) == true

    @test isonehot([0.03, 0.03, 0.94]) == false
    @test isonehot([0.03, 0.94, 0.03]) == false
    @test isonehot([0.94, 0.03, 0.03]) == false

    @test isonehot([0.0003, 0.0003, 0.9994]) == false
    @test isonehot([0.0003, 0.9994, 0.0003]) == false
    @test isonehot([0.9994, 0.0003, 0.0003]) == false
end
