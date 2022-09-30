module AlgebraCommonTest

using Test
using ReactiveMP
using Random
using LinearAlgebra

@testset "Common" begin
    @testset "mirrorlog" begin
        rng  = MersenneTwister(1234)
        vals = rand(rng, 10)
        @test ReactiveMP.mirrorlog.(vals) == map(x -> log(1 - x), vals)
    end

    @testset "xtlog" begin
        rng  = MersenneTwister(1234)
        vals = rand(rng, 10)
        @test ReactiveMP.xtlog.(vals) == map(x -> x * log(x), vals)
    end

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

            # See: https://github.com/JuliaSIMD/LoopVectorization.jl/issues/377
            # Remove if statement once fixed
            if T1 !== Float32 || T2 !== Float32
                @test ReactiveMP.mul_trace(A, B) ≈ tr(A * B)
            end

            a = rand(rng, T1)
            b = rand(rng, T2)
            @test ReactiveMP.mul_trace(a, b) ≈ a * b
        end
    end

    @testset "tiny/huge" begin
        import ReactiveMP: TinyNumber, HugeNumber

        @test typeof(tiny) === TinyNumber
        @test typeof(huge) === HugeNumber

        @test convert(Float32, tiny) == 1.0f-6
        @test convert(Float64, tiny) == 1e-12
        @test convert(BigFloat, tiny) == big"1e-24"

        @test Float32(tiny) == 1.0f-6
        @test Float64(tiny) == 1e-12
        @test BigFloat(tiny) == big"1e-24"

        @test convert(Float32, huge) == 1.0f+6
        @test convert(Float64, huge) == 1e+12
        @test convert(BigFloat, huge) == big"1e+24"

        @test Float32(huge) == 1.0f+6
        @test Float64(huge) == 1e+12
        @test BigFloat(huge) == big"1e+24"

        @test @inferred clamp(1.0f0, tiny, huge) == 1.0f0
        @test @inferred clamp(0.0f0, tiny, huge) == 1.0f-6
        @test @inferred clamp(1.0f13, tiny, huge) == 1.0f+6

        @test @inferred clamp(1.0, tiny, huge) == 1.0
        @test @inferred clamp(0.0, tiny, huge) == 1e-12
        @test @inferred clamp(1e13, tiny, huge) == 1e12

        @test @inferred clamp(big"1.0", tiny, huge) == big"1.0"
        @test @inferred clamp(big"0.0", tiny, huge) == big"1e-24"
        @test @inferred clamp(big"1e25", tiny, huge) == big"1e+24"

        for a in (1, 1.0, 0, 0.0, 1.0f0, 0.0f0, Int32(0), Int32(1), big"1", big"1.0", big"0", big"0.0")
            T = typeof(a)
            for v in [tiny, huge]
                V = typeof(v)

                for op in [+, -, *, /, >, >=, <, <=]
                    @test @inferred op(a, v) == op(a, convert(promote_type(T, V), v))
                    @test @inferred op(v, a) == op(convert(promote_type(T, V), v), a)

                    @test @inferred op(a, v) == op(a, v)
                    @test @inferred op(v, a) == op(v, a)
                end

                @test v <= (@inferred clamp(a, v, Inf)) <= Inf
                @test zero(a) <= (@inferred clamp(a, zero(a), v)) <= v

                for size in [3, 5]
                    for array in [fill(a, (size,)), fill(a, (size, size))]
                        for op in [+, -, *, /, >, >=, <, <=]
                            @test @inferred op.(array, v) == op.(array, convert(promote_type(T, V), v))
                            @test @inferred op.(v, array) == op.(convert(promote_type(T, V), v), array)

                            @test @inferred op.(array, v) == op.(array, v)
                            @test @inferred op.(v, array) == op.(v, array)
                        end

                        @test @inferred clamp.(array, v, Inf) == clamp.(array, v, Inf)
                        @test @inferred clamp.(array, zero(array), v) == clamp.(array, zero(array), v)
                    end
                end
            end
        end
    end
end

end
