module ReactiveMPTestingHelpers

using Test
using ReactiveMP

import ReactiveMP: SkipIndexIterator, skipindex
import ReactiveMP: clamplog, deep_eltype
import ReactiveMP: InfCountingReal, ∞
import ReactiveMP: FunctionalIndex

@testset "Helpers" begin

    @testset "SkipIndexIterator" begin
        s = skipindex(1:3, 2)
        @test typeof(s) <: SkipIndexIterator
        @test collect(s) == [1, 3]
        @test collect(skipindex(s, 1)) == [3]
    end

    @testset "clamplog" begin
        @test !isnan(clamplog(0.0)) && !isinf(clamplog(0.0))
        @test clamplog(tiny + 1.0) === log(tiny + 1.0)
    end

    @testset "deep_eltype" begin
        for type in [Float32, Float64, Complex{Float64}, BigFloat]
            @test deep_eltype(type) === type
            @test deep_eltype(zero(type)) === type

            vector             = zeros(type, 10)
            matrix             = zeros(type, 10, 10)
            vector_of_vectors  = [vector, vector]
            vector_of_matrices = [matrix, matrix]
            matrix_of_vector   = [vector vector; vector vector]
            matrix_of_matrices = [matrix matrix; matrix matrix]

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

    @testset "FunctionalIndex" begin
        for N in 1:5
            collection = ones(N)
            @test FunctionalIndex{:nothing}(firstindex)(collection) === firstindex(collection)
            @test FunctionalIndex{:nothing}(lastindex)(collection) === lastindex(collection)
            @test (FunctionalIndex{:nothing}(firstindex) + 1)(collection) === firstindex(collection) + 1
            @test (FunctionalIndex{:nothing}(lastindex) - 1)(collection) === lastindex(collection) - 1
            @test (FunctionalIndex{:nothing}(firstindex) + 1 - 2 + 3)(collection) === firstindex(collection) + 1 - 2 + 3
            @test (FunctionalIndex{:nothing}(lastindex) - 1 + 2 - 3)(collection) === lastindex(collection) - 1 + 2 - 3
        end

        @test repr(FunctionalIndex{:begin}(firstindex)) === "(begin)"
        @test repr(FunctionalIndex{:begin}(firstindex) + 1) === "((begin) + 1)"
        @test repr(FunctionalIndex{:begin}(firstindex) - 1) === "((begin) - 1)"
        @test repr(FunctionalIndex{:begin}(firstindex) - 1 + 1) === "(((begin) - 1) + 1)"

        @test repr(FunctionalIndex{:end}(lastindex)) === "(end)"
        @test repr(FunctionalIndex{:end}(lastindex) + 1) === "((end) + 1)"
        @test repr(FunctionalIndex{:end}(lastindex) - 1) === "((end) - 1)"
        @test repr(FunctionalIndex{:end}(lastindex) - 1 + 1) === "(((end) - 1) + 1)"

        @test isbitstype(typeof((FunctionalIndex{:begin}(firstindex) + 1)))
        @test isbitstype(typeof((FunctionalIndex{:begin}(firstindex) - 1)))
        @test isbitstype(typeof((FunctionalIndex{:begin}(firstindex) + 1 + 1)))
        @test isbitstype(typeof((FunctionalIndex{:begin}(firstindex) - 1 + 1)))
    end
end

end
