module AlgebraCholeskyTest

using Test
using ReactiveMP
using Random

using LinearAlgebra

@testset "Helpers" begin
    @testset "cholesky related" begin
        rng = MersenneTwister(1234)

        for scalar in (1, 1.0, 2.0, 2, 0.1, 10.0)
            A = scalar * I

            @test A isa UniformScaling
            @test ReactiveMP.cholinv(A) ≈ inv(scalar) * I
            @test ReactiveMP.cholsqrt(A) ≈ sqrt(scalar) * I

            # By default `logdet` is defined only for `λ = 0 or 1`
            if isone(scalar) || iszero(scalar)
                @test ReactiveMP.chollogdet(A) ≈ logdet(A)
            else
                @test_throws ArgumentError ReactiveMP.chollogdet(A)
            end
        end

        for size in (2, 3, 4, 5, 10, 100, 1000)
            L = rand(rng, size, size)
            A = L * L'

            @test LinearAlgebra.cholesky(A).L ≈ ReactiveMP.fastcholesky(A).L
            @test inv(LinearAlgebra.cholesky(A)) ≈ inv(ReactiveMP.fastcholesky(A))
            @test inv(LinearAlgebra.cholesky(A)) ≈ ReactiveMP.cholinv(A)
            @test sqrt(A) * sqrt(A)' ≈ ReactiveMP.cholsqrt(A) * ReactiveMP.cholsqrt(A)'
            @test A ≈ ReactiveMP.cholsqrt(A) * ReactiveMP.cholsqrt(A)'
            @test logdet(A) ≈ ReactiveMP.chollogdet(A)
            @test all((inv(cholesky(A)), logdet(A)) .≈ ReactiveMP.cholinv_logdet(A))
        end

        for size in (2, 3, 4, 5, 10, 100, 1000)
            A = Diagonal(rand(rng, size))
            @test inv(A) ≈ ReactiveMP.cholinv(A)
            @test (sqrt(A) * sqrt(A)') ≈ (ReactiveMP.cholsqrt(A) * ReactiveMP.cholsqrt(A)')
            @test A ≈ (ReactiveMP.cholsqrt(A) * ReactiveMP.cholsqrt(A)')
            @test logdet(A) ≈ ReactiveMP.chollogdet(A)
            @test all((inv(A), logdet(A)) .≈ ReactiveMP.cholinv_logdet(A))
        end

        for A in rand(rng, 3)
            @test inv(A) ≈ ReactiveMP.cholinv(A)
            @test (sqrt(A) * sqrt(A)') ≈ (ReactiveMP.cholsqrt(A) * ReactiveMP.cholsqrt(A)')
            @test A ≈ (ReactiveMP.cholsqrt(A) * ReactiveMP.cholsqrt(A)')
            @test logdet(A) ≈ ReactiveMP.chollogdet(A)
            @test all((inv(A), logdet(A)) .≈ ReactiveMP.cholinv_logdet(A))
        end
    end
end

end
