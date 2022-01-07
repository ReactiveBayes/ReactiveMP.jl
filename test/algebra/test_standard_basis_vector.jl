module ReactiveMPStandardBasisVectorTest

using Test
using ReactiveMP 
using Random

using LinearAlgebra

@testset "StandardBasisVector" begin

    rng = MersenneTwister(1234)

    @testset begin 
        @test_throws AssertionError StandardBasisVector(0, 1)
        @test_throws AssertionError StandardBasisVector(-10, 1)
        @test_throws AssertionError StandardBasisVector(10, 11)
        @test_throws AssertionError StandardBasisVector(10, -2)

        for T in (Int, Float64, Float32)
            r = rand(rng, T)
            e = StandardBasisVector(2, 1, r)
            @test eltype(e) === T
            @test e[1] === r
            @test e[2] === zero(r)
        end
    end

    # Same sizes
    @testset begin 

        for N in 1:8
            for I in 1:N
                for T in (Int, Float64, Float32)
                    scale  = rand(rng, T)
                    e      = StandardBasisVector(N, I, scale)
                    e_c    = zeros(T, N)
                    e_c[I] = scale

                    m = rand(rng, T)
                    v = rand(rng, T, N)
                    A = rand(rng, T, N, N)
                    a = rand(rng, T, N, 1)

                    @test m * e == m * e_c
                    @test m * e' == m * e_c'
                    @test e * m == e_c * m
                    @test e' * m == e_c' * m

                    @test (A * e) == (A * e_c)
                    @test (A' * e) == (A' * e_c)
                    @test (e * e') == (e_c * e_c') 
                    @test (e' * e) == (e_c' * e_c)
                    @test (v' * e) == (v' * e_c)
                    @test (e' * v) == (e_c' * v)
                    @test (e' * v) == (e_c' * v)
                    @test (a' * e) == (a' * e_c)
                    @test (a * e') == (a * e_c')

                    t = rand(rng, T)

                    @test ReactiveMP.v_a_vT(e, t)    ≈ ReactiveMP.v_a_vT(e_c, t)
                    @test ReactiveMP.v_a_vT(e, t, e) ≈ ReactiveMP.v_a_vT(e_c, t, e_c)

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
        end

    # Different sizes
    @testset begin 

        for N1 in 1:4, N2 in 1:4
            if N1 !== N2
                for I1 in 1:N1, I2 in 1:N2
                    for T in (Int, Float64, Float32)
                        scale1   = rand(rng, T)
                        scale2   = rand(rng, T)
                        e1       = StandardBasisVector(N1, I1, scale1)
                        e2       = StandardBasisVector(N2, I2, scale2)
                        e_c1     = zeros(T, N1)
                        e_c1[I1] = scale1
                        e_c2     = zeros(T, N2)
                        e_c2[I2] = scale2

                        @test_throws AssertionError dot(e1, e2)
                        @test_throws AssertionError dot(e_c1, e2)
                        @test_throws AssertionError dot(e1, e_c2)
                        @test_throws AssertionError dot(e2, e1)
                        @test_throws AssertionError dot(e_c2, e1)
                        @test_throws AssertionError dot(e2, e_c1)

                        @test e1 * e2' == e_c1 * e_c2'
                        @test e2 * e1' == e_c2 * e_c1'
                        @test e_c1 * e2' == e_c1 * e_c2'
                        @test e_c2 * e1' == e_c2 * e_c1'
                        @test e1 * e_c2' == e_c1 * e_c2'
                        @test e2 * e_c1' == e_c2 * e_c1'
                        @test_throws AssertionError e1' * e2
                        @test_throws AssertionError e1' * e_c2
                        @test_throws AssertionError e2' * e1
                        @test_throws AssertionError e2' * e_c1
                    end
                end
            end
        end

    end
    
end

end