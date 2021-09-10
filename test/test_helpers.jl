module ReactiveMPTestingHelpers

using Test
using ReactiveMP

import ReactiveMP: OneDivNVector
import ReactiveMP: deep_eltype

export enabled_tests, addtests, key_to_filename, filename_to_key

enabled_tests = lowercase.(ARGS)

function addtests(filename)
    key = filename_to_key(filename)
    if isempty(enabled_tests) || key in enabled_tests
        include(filename)
    end
end

function key_to_filename(key)
    splitted = split(key, ":")
    return length(splitted) === 1 ? string("test_", first(splitted), ".jl") : string(join(splitted[1:end - 1], "/"), "/test_", splitted[end], ".jl")
end

function filename_to_key(filename)
    splitted   = split(filename, "/")
    if length(splitted) === 1
        return replace(replace(first(splitted), ".jl" => ""), "test_" => "")
    else
        path, name = splitted[1:end - 1], splitted[end]
        return string(join(path, ":"), ":", replace(replace(name, ".jl" => ""), "test_" => ""))
    end
end

@testset "Helpers" begin
    
    @testset "OneDivNVector" begin 

        for type in [ Float64, Float32, BigFloat ], len in [ 3, 5, 10 ]
            iter = OneDivNVector(type, len)

            @test eltype(iter) === type
            @test length(iter) === len
            @test size(iter)   === (len, )
            @test collect(iter) == fill(one(type) / len, len)
            @test vec(iter)     == fill(one(type) / len, len)
            @test sizeof(iter) === 0

            sim = similar(iter)

            @test sim === iter
            @test eltype(sim) === type
            @test length(sim) === len
            @test size(sim)   === (len, )
            @test collect(sim) == fill(one(type) / len, len)
            @test vec(sim)     == fill(one(type) / len, len)
            @test sizeof(sim) === 0

            sim = similar(iter, Float16)
            @test eltype(sim) === Float16
            @test length(sim) === len
            @test size(sim)   === (len, )
            @test collect(sim) == fill(one(Float16) / len, len)
            @test vec(sim)     == fill(one(Float16) / len, len)
            @test sizeof(sim) === 0
        end

        @test eltype(OneDivNVector(3)) === Float64
        @test eltype(OneDivNVector(5)) === Float64

        @test_throws AssertionError OneDivNVector(-2)
        @test_throws AssertionError OneDivNVector(-10)
        @test_throws AssertionError OneDivNVector(Vector, 2)
        @test_throws AssertionError OneDivNVector(Matrix, 10)

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
    
end

end