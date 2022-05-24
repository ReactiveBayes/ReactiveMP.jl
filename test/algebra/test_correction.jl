module ReactiveMPCorrectionTest

using Test
using ReactiveMP
using Random

using LinearAlgebra

@testset "Correction" begin
    rng = MersenneTwister(1234)

    for n in [3, 5, 10]
        a = rand(rng)
        A = rand(rng, n, n)

        b = ReactiveMP.correction!(NoCorrection(), a)
        B = ReactiveMP.correction!(NoCorrection(), A)

        @test a == b
        @test a === b
        @test A == B
        @test A === B

        c = ReactiveMP.correction!(TinyCorrection(), ReactiveMP.tiny / 2)
        @test c >= ReactiveMP.tiny

        C = ReactiveMP.correction!(TinyCorrection(), copy(A))

        @test A ≈ C
        @test mapreduce((d) -> d[1] + ReactiveMP.tiny === d[2], &, zip(diag(A), diag(C)))

        D = rand(rng, n, n)
        E = ReactiveMP.correction!(TinyCorrection(), D)
        @test D === E

        v = 1e-10 * rand(rng)

        f = ReactiveMP.correction!(FixedCorrection(v), v / 2)
        @test f >= v

        F = ReactiveMP.correction!(FixedCorrection(v), copy(A))
        @test A ≈ F
        @test mapreduce((d) -> d[1] + v === d[2], &, zip(diag(A), diag(F)))

        G = rand(rng, n, n)
        H = ReactiveMP.correction!(FixedCorrection(v), G)
        @test G === H

        j = ReactiveMP.correction!(ClampEigenValuesCorrection(10.0), 5.0)
        @test j >= 10.0

        J   = ReactiveMP.correction!(ClampEigenValuesCorrection(10.0), copy(A))
        S_J = svd(J)

        @test mapreduce((d) -> d >= 10.0 || d ≈ 10.0, &, S_J.S)

        K = ReactiveMP.correction!(ClampEigenValuesCorrection(1e-12), copy(A))
        @test K ≈ A
    end
end

end
