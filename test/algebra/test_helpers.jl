module AlgebraHelpersTest

using Test
using ReactiveMP 
using Random

using LinearAlgebra

@testset "Helpers" begin

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
            @test ReactiveMP.rank1update(A, x) ≈ (A + x*x')
            y = rand(rng, T3, size)
            @test ReactiveMP.rank1update(A, x, y) ≈ (A + x*y')
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
    
end

end