module ReactiveMPStandardBasisVectorTest

using Test
using ReactiveMP 
using Random

using LinearAlgebra

@testset "StandardBasisVector" begin

    rng = MersenneTwister(1234)

    @testset begin 

        for N in 1:8
            for I in 1:N
                for T in (Int, Float64, Float32)
                    scale  = rand(rng, T)
                    e      = StandardBasisVector(N, I, scale)
                    e_c    = zeros(T, N)
                    e_c[I] = scale

                    v = rand(rng, T, N)
                    A = rand(rng, T, N, N)
                    a = rand(rng, T, N, 1)

                    @test eltype(A * e) === eltype(scale)
                    @test eltype(A' * e) === eltype(scale)
                    @test eltype(e * e') === eltype(scale)
                    @test eltype(e' * e) === eltype(scale)
                    @test eltype(v' * e) === eltype(scale)
                    @test eltype(e * v') === eltype(scale)
                    @test eltype(a' * e) === eltype(scale)

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

    end
    
end

end