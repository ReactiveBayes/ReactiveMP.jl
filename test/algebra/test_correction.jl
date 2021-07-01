module ReactiveMPCorrectionTest

using Test
using ReactiveMP 
using Random

using LinearAlgebra

@testset "Correction" begin

    rng = MersenneTwister(1234)

    for n in [ 3, 5, 10 ]
        A = rand(rng, n, n)

        B = ReactiveMP.correction!(NoCorrection(), A)

        @test A == B
        @test A === B

        C = ReactiveMP.correction!(TinyCorrection(), copy(A))

        @test A ≈ C
        @test mapreduce((d) -> d[1] + ReactiveMP.tiny === d[2], &, zip(diag(A), diag(C)))

        D = rand(rng, n, n)
        E = ReactiveMP.correction!(TinyCorrection(), D)
        @test D === E
    end
    
end

end