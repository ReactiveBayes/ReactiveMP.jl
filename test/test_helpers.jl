module ReactiveMPTestingHelpers

using Test
using ReactiveMP

import ReactiveMP: SkipIndexIterator, skipindex
import ReactiveMP: deep_eltype
import ReactiveMP: InfCountingReal, ∞

@testset "Helpers" begin

    @testset "SkipIndexIterator" begin  
        s = skipindex(1:3, 2)
        @test typeof(s) <: SkipIndexIterator
        @test collect(s) == [ 1, 3 ]
        @test collect(skipindex(s, 1)) == [ 3 ]
    end

    @testset "deep_eltype" begin

        for type in [ Float32, Float64, Complex{Float64}, BigFloat ]

            @test deep_eltype(type) === type
            @test deep_eltype(zero(type)) === type

            vector = zeros(type, 10)
            matrix = zeros(type, 10, 10)
            vector_of_vectors  = [ vector, vector ]
            vector_of_matrices = [ matrix, matrix ]
            matrix_of_vector   = [ vector vector; vector vector ]
            matrix_of_matrices = [ matrix matrix; matrix matrix ]

            @test deep_eltype(vector) === type
            @test deep_eltype(matrix) === type
            @test deep_eltype(vector_of_vectors) === type
            @test deep_eltype(vector_of_matrices) === type
            @test deep_eltype(matrix_of_vector) === type
            @test deep_eltype(matrix_of_matrices) === type

        end

    end

    @testset "InfCountingReal" begin 
        r = InfCountingReal(0.0, 0)
        @test float(r) ≈ 0.0
        @test float(r + 1) ≈ 1.0
        @test float(1 + r) ≈ 1.0
        @test float(r - 1) ≈ -1.0
        @test float(1 - r) ≈ 1.0
        @test float(r - 1 + ∞) ≈ Inf
        @test float(1 - r + ∞) ≈ Inf
        @test float(r - 1 + ∞ - ∞) ≈ -1.0
        @test float(1 - r + ∞ - ∞) ≈ 1.0
    end
    
end

end