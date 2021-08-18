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

    @testset "Statistics" begin

        rng = MersenneTwister(42)

        # All
        for N in [ 5, 10, 100 ]

            scalar_samples = rand(rng, N)
            scalar_weights = rand(rng, N)
            scalar_samplelist = SampleList(scalar_samples, scalar_weights)

            @test mean(scalar_samplelist)    ≈ sum(scalar_weights .* scalar_samples)
            @test logmean(scalar_samplelist) ≈ sum(scalar_weights .* log.(scalar_samples))
            @test meanlogmean(scalar_samplelist) ≈ sum(scalar_weights .* scalar_samples .* log.(scalar_samples))

            vector_samples = [ rand(rng, 2) for _ in 1:N ]
            vector_weights = rand(rng, N)
            vector_samplelist = SampleList(vector_samples, vector_weights)

            @test mean(vector_samplelist)    ≈ sum(vector_weights .* vector_samples)
            @test logmean(vector_samplelist) ≈ sum(vector_weights .* map(e -> log.(e), (vector_samples)))
            @test meanlogmean(vector_samplelist) ≈ sum(vector_weights .* map(e -> e .* log.(e), (vector_samples)))

            matrix_samples = [ rand(rng, 2, 2) for _ in 1:N ]
            matrix_weights = rand(rng, N)
            matrix_samplelist = SampleList(matrix_samples, matrix_weights)

            @test mean(matrix_samplelist) ≈ sum(matrix_weights .* matrix_samples)
            @test logmean(matrix_samplelist) ≈ sum(matrix_weights .* map(e -> log.(e), matrix_samples))
            @test meanlogmean(matrix_samplelist) ≈ sum(matrix_weights .* map(e -> e .* log.(e), matrix_samples))

        end

        # Logmean and meanlogmean
        uni_distribution = Gamma(rand(rng) + 1, rand(rng) + 2)
        uni_samples      = rand(rng, uni_distribution, 20_000)
        uni_sample_list  = SampleList(uni_samples)

        r = rand(rng, 3)
        Σ = diageye(3) + 2r*r' # positive definite matrix
        mv_distribution = MvNormal(r, Σ)
        mv_samples      = [ rand(rng, mv_distribution) for _ in 1:20_000 ]
        mv_sample_list  = SampleList(mv_samples)

        r = rand(rng, 2)
        W = diageye(2) + 2r*r' # positive definite matrix
        mxv_distribution = Wishart(3, W)
        mxv_samples      = [ rand(rng, mxv_distribution) for _ in 1:20_000 ]
        mxv_sample_list  = SampleList(mxv_samples)

        @test isapprox(mean(uni_sample_list), mean(uni_distribution), atol = 0.1)
        @test isapprox(mean(mv_sample_list), mean(mv_distribution), atol = 0.1)
        @test isapprox(mean(mxv_sample_list), mean(mxv_distribution), atol = 1.0)

        @test isapprox(var(uni_sample_list), var(uni_distribution), atol = 0.5)
        @test isapprox(cov(uni_sample_list), var(uni_distribution), atol = 0.5)
        @test isapprox(var(mv_sample_list), var(mv_distribution), atol = 0.1)
        @test isapprox(cov(mv_sample_list), cov(mv_distribution), atol = 0.1)
        @test_broken isapprox(var(mxv_sample_list), var(mxv_distribution), atol = 0.1)
        @test_broken isapprox(cov(mxv_sample_list), cov(mxv_distribution), atol = 0.1)

        @test isapprox(std(uni_sample_list), std(uni_distribution), atol = 0.2)
        @test isapprox(std(mv_sample_list), cholsqrt(cov(mv_distribution)), atol = 0.2)

        @test isapprox(invcov(uni_sample_list), inv(var(uni_distribution)), atol = 0.2)
        @test isapprox(invcov(mv_sample_list), cholinv(cov(mv_distribution)), atol = 0.2)

        # TODO logmean for matrix variate distribution?

        mv_distribution = Dirichlet(rand(rng, 3))
        mv_samples      = [ rand(rng, mv_distribution) for _ in 1:20_000 ]
        mv_sample_list  = SampleList(mv_samples)

        @test isapprox(logmean(uni_sample_list), logmean(uni_distribution); atol = 0.02)
        @test isapprox(logmean(mv_sample_list), logmean(mv_distribution); atol = 0.5)

        # TODO meanlogmean for multivariate and matrix variate distribution?

        @test isapprox(meanlogmean(uni_sample_list), meanlogmean(uni_distribution); atol = 0.25)


        mxv_sample_list = SampleList([ diageye(2), diageye(2) ], [ 0.5, 0.5 ])
        @test_broken cov(mxv_sample_list) == zeros(4, 4)

    end

    @testset "vague" begin 

        @test variate_form(vague(SampleList))    === Univariate
        @test variate_form(vague(SampleList, 2)) === Multivariate
        @test variate_form(vague(SampleList, 2, 2)) === Matrixvariate
        @test variate_form(vague(SampleList, (3, 4))) === Matrixvariate

        @test ndims(vague(SampleList))    === 1
        @test ndims(vague(SampleList, 2)) === 2
        @test ndims(vague(SampleList, 2, 2)) === (2, 2)
        @test ndims(vague(SampleList, (3, 4))) === (3, 4)

        for nsamples in [ 10, 100, 1000 ]
            @test variate_form(vague(SampleList; nsamples = nsamples))    === Univariate
            @test variate_form(vague(SampleList, 2; nsamples = nsamples)) === Multivariate
            @test variate_form(vague(SampleList, 2, 2; nsamples = nsamples)) === Matrixvariate
            @test variate_form(vague(SampleList, (3, 4); nsamples = nsamples)) === Matrixvariate

            @test length(vague(SampleList; nsamples = nsamples))    === nsamples
            @test length(vague(SampleList, 2; nsamples = nsamples)) === nsamples
            @test length(vague(SampleList, 2, 2; nsamples = nsamples)) === nsamples
            @test length(vague(SampleList, (3, 4); nsamples = nsamples)) === nsamples

            @test ndims(vague(SampleList; nsamples = nsamples))    === 1
            @test ndims(vague(SampleList, 2; nsamples = nsamples)) === 2
            @test ndims(vague(SampleList, 2, 2; nsamples = nsamples)) === (2, 2)
            @test ndims(vague(SampleList, (3, 4); nsamples = nsamples)) === (3, 4)

            @test size(vague(SampleList; nsamples = nsamples))    === (nsamples, )
            @test size(vague(SampleList, 2; nsamples = nsamples)) === (nsamples, )
            @test size(vague(SampleList, 2, 2; nsamples = nsamples)) === (nsamples, )
            @test size(vague(SampleList, (3, 4); nsamples = nsamples)) === (nsamples, )
        end

    end

end

end
