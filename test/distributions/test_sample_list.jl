module SampleListTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: deep_eltype, getsamples, getweights

@testset "SampleList" begin

    @testset "Constructor" begin
        
        rng = MersenneTwister(1234)

        for N in [ 5, 10, 100 ], type in [ Float64, Float32, BigFloat ]
            scalar_samples    = rand(rng, type, N)
            scalar_samplelist = SampleList(scalar_samples)

            @test collect(getsamples(scalar_samplelist)) == scalar_samples
            @test collect(getweights(scalar_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(getsamples(scalar_samplelist)) === type
            @test eltype(getweights(scalar_samplelist))      === type
            @test variate_form(scalar_samplelist) === Univariate

            scalar_weights    = rand(rng, type, N)
            scalar_samplelist = SampleList(scalar_samples, scalar_weights)

            @test collect(getsamples(scalar_samplelist)) == scalar_samples
            @test collect(getweights(scalar_samplelist)) == scalar_weights
            @test deep_eltype(getsamples(scalar_samplelist)) === type
            @test eltype(getweights(scalar_samplelist))      === type
            @test variate_form(scalar_samplelist) === Univariate

            vector_samples    = [ rand(rng, type, 2) for _ in 1:N ]
            vector_samplelist = SampleList(vector_samples)

            @test collect(getsamples(vector_samplelist)) == vector_samples
            @test collect(getweights(vector_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(getsamples(vector_samplelist)) === type
            @test eltype(getweights(vector_samplelist))      === type
            @test variate_form(vector_samplelist) === Multivariate

            vector_weights    = rand(rng, type, N)
            vector_samplelist = SampleList(vector_samples, vector_weights)

            @test collect(getsamples(vector_samplelist)) == vector_samples
            @test collect(getweights(vector_samplelist)) == vector_weights
            @test deep_eltype(getsamples(vector_samplelist)) === type
            @test eltype(getweights(vector_samplelist))      === type
            @test variate_form(vector_samplelist) === Multivariate

            matrix_samples    = [ rand(rng, type, 2, 2) for _ in 1:N ]
            matrix_samplelist = SampleList(matrix_samples)

            @test collect(getsamples(matrix_samplelist)) == matrix_samples
            @test collect(getweights(matrix_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(getsamples(matrix_samplelist)) === type
            @test eltype(getweights(matrix_samplelist))      === type
            @test variate_form(matrix_samplelist) === Matrixvariate

            matrix_weights    = rand(rng, type, N)
            matrix_samplelist = SampleList(matrix_samples, matrix_weights)

            @test collect(getsamples(matrix_samplelist)) == matrix_samples
            @test collect(getweights(matrix_samplelist)) == matrix_weights
            @test deep_eltype(getsamples(matrix_samplelist)) === type
            @test eltype(getweights(matrix_samplelist))      === type
            @test variate_form(matrix_samplelist) === Matrixvariate
        end

        @test_throws AssertionError SampleList(rand(10), rand(5))
        @test_throws AssertionError SampleList(rand(5), rand(10))
        @test_throws AssertionError SampleList(rand(5), [ nothing for _ in 1:5 ])

        @test_throws AssertionError SampleList([ rand(10) for _ in 1:10 ], rand(5))
        @test_throws AssertionError SampleList([ rand(5) for _ in 1:5 ], rand(10))
        @test_throws AssertionError SampleList([ rand(5) for _ in 1:5 ], [ nothing for _ in 1:5 ])

        @test_throws AssertionError SampleList([ rand(10, 10) for _ in 1:10 ], rand(5))
        @test_throws AssertionError SampleList([ rand(5, 5) for _ in 1:5 ], rand(10))
        @test_throws AssertionError SampleList([ rand(5, 5) for _ in 1:5 ], [ nothing for _ in 1:5 ])

    end

end

end
