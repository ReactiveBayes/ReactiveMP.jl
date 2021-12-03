module ReactiveMPTestingHelpers

using Test
using ReactiveMP

import ReactiveMP: OneDivNVector
import ReactiveMP: deep_eltype

@testset "Helpers" begin

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
    
end

end