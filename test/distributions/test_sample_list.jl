module SampleListTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: deep_eltype, get_samples, get_weights, sample_list_zero_element
import ReactiveMP: get_meta, get_unnormalised_weights, get_entropy, get_logproposal, get_logintegrand
import ReactiveMP: call_logproposal, call_logintegrand


@testset "SampleList" begin

    @testset "Internal functions" begin
        @test sample_list_zero_element(SampleList([ 1.0, 1.0 ])) === 0.0
        @test sample_list_zero_element(SampleList([ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ])) == [ 0.0, 0.0 ]
        @test sample_list_zero_element(SampleList([ [ 1.0 1.0; 1.0 1.0 ], [ 1.0 1.0; 1.0 1.0 ] ])) == [ 0.0 0.0; 0.0 0.0 ]
        @test sample_list_zero_element(SampleList([ [ 1.0 1.0 1.0; 1.0 1.0 1.0 ], [ 1.0 1.0 1.0; 1.0 1.0 1.0 ] ])) == [ 0.0 0.0 0.0; 0.0 0.0 0.0 ]
        @test sample_list_zero_element(SampleList([ [ 1.0; 1.0], [ 1.0; 1.0 ] ])) == [ 0.0; 0.0 ]
    end

    @testset "Constructor" begin
        
        rng = MersenneTwister(1234)

        for N in [ 5, 10, 100 ], type in [ Float64, Float32, BigFloat ]
            scalar_samples    = rand(rng, type, N)
            scalar_samplelist = SampleList(scalar_samples)

            @test collect(get_samples(scalar_samplelist)) == scalar_samples
            @test collect(get_weights(scalar_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(scalar_samplelist)              === type
            @test eltype(scalar_samplelist)                   === type
            @test eltype(get_weights(scalar_samplelist))      === type
            @test variate_form(scalar_samplelist) === Univariate

            scalar_weights    = rand(rng, type, N)
            scalar_samplelist = SampleList(scalar_samples, scalar_weights)

            @test collect(get_samples(scalar_samplelist)) == scalar_samples
            @test collect(get_weights(scalar_samplelist)) == scalar_weights
            @test deep_eltype(scalar_samplelist)              === type
            @test eltype(scalar_samplelist)                   === type
            @test eltype(get_weights(scalar_samplelist))      === type
            @test variate_form(scalar_samplelist) === Univariate

            vector_samples    = [ rand(rng, type, 2) for _ in 1:N ]
            vector_samplelist = SampleList(vector_samples)

            @test collect(get_samples(vector_samplelist)) == vector_samples
            @test collect(get_weights(vector_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(vector_samplelist)              === type
            @test eltype(vector_samplelist)                   === Vector{ type }
            @test eltype(get_weights(vector_samplelist))      === type
            @test variate_form(vector_samplelist) === Multivariate

            vector_weights    = rand(rng, type, N)
            vector_samplelist = SampleList(vector_samples, vector_weights)

            @test collect(get_samples(vector_samplelist)) == vector_samples
            @test collect(get_weights(vector_samplelist)) == vector_weights
            @test deep_eltype(vector_samplelist)              === type
            @test eltype(vector_samplelist)                   === Vector{ type }
            @test eltype(get_weights(vector_samplelist))      === type
            @test variate_form(vector_samplelist) === Multivariate

            matrix_samples    = [ rand(rng, type, 2, 2) for _ in 1:N ]
            matrix_samplelist = SampleList(matrix_samples)

            @test collect(get_samples(matrix_samplelist)) == matrix_samples
            @test collect(get_weights(matrix_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(matrix_samplelist)              === type
            @test eltype(matrix_samplelist)                   === Matrix{ type }
            @test eltype(get_weights(matrix_samplelist))      === type
            @test variate_form(matrix_samplelist) === Matrixvariate

            matrix_weights    = rand(rng, type, N)
            matrix_samplelist = SampleList(matrix_samples, matrix_weights)

            @test collect(get_samples(matrix_samplelist)) == matrix_samples
            @test collect(get_weights(matrix_samplelist)) == matrix_weights
            @test deep_eltype(matrix_samplelist)              === type
            @test eltype(matrix_samplelist)                   === Matrix{ type }
            @test eltype(get_weights(matrix_samplelist))      === type
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

        uni_distribution2 = NormalMeanVariance(rand(rng) + 1, rand(rng) + 2)
        uni_samples2      = rand(rng, uni_distribution2, 20_000)
        uni_sample_list2  = SampleList(uni_samples2)

        r = rand(rng, 3)
        Σ = diageye(3) + 2r*r' # positive definite matrix
        mv_distribution = MvNormal(r, Σ)
        mv_samples      = [ rand(rng, mv_distribution) for _ in 1:20_000 ]
        mv_sample_list  = SampleList(mv_samples)

        r1 = rand(rng, 3)
        W1 = rand(3, 4)
        r2 = rand(rng, 3)
        W2 = diageye(3) + 2r2*r2' # positive definite matrix
        r3 = rand(rng, 4)
        W3 = diageye(4) + 2r3*r3' # positive definite matrix
        mxv_distribution = MatrixNormal(W1, W2, W3)
        mxv_samples      = [ rand(rng, mxv_distribution) for _ in 1:20_000 ]
        mxv_sample_list  = SampleList(mxv_samples)

        @test isapprox(mean(uni_sample_list), mean(uni_distribution), atol = 0.1)
        @test isapprox(mean(mv_sample_list), mean(mv_distribution), atol = 0.1)
        @test isapprox(mean(mxv_sample_list), mean(mxv_distribution), atol = 1.0)

        @test isapprox(weightedmean(uni_sample_list2), weightedmean(uni_distribution2), atol = 0.1)
        @test all(isapprox.(mean_var(uni_sample_list2), mean_var(uni_distribution2), atol = 0.1))
        @test all(isapprox.(mean_cov(uni_sample_list2), mean_cov(uni_distribution2), atol = 0.1))
        @test all(isapprox.(mean_precision(uni_sample_list2), mean_precision(uni_distribution2), atol = 0.1))
        @test all(isapprox.(mean_invcov(uni_sample_list2), mean_invcov(uni_distribution2), atol = 0.1))
        @test all(isapprox.(weightedmean_cov(uni_sample_list2), weightedmean_cov(uni_distribution2), atol = 0.1))
        @test all(isapprox.(weightedmean_invcov(uni_sample_list2), weightedmean_invcov(uni_distribution2), atol = 0.1))
        @test all(isapprox.(weightedmean_precision(uni_sample_list2), weightedmean_precision(uni_distribution2), atol = 0.1))   

        @test isapprox(var(uni_sample_list), var(uni_distribution), atol = 0.5)
        @test isapprox(cov(uni_sample_list), var(uni_distribution), atol = 0.5)
        @test isapprox(var(mv_sample_list), var(mv_distribution), atol = 0.1)
        @test isapprox(cov(mv_sample_list), cov(mv_distribution), atol = 0.1)
        @test isapprox(var(mxv_sample_list), var(mxv_distribution), atol = 0.1)
        @test isapprox(cov(mxv_sample_list), cov(mxv_distribution), atol = 1.0)

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

        r4 = rand(rng, 5)
        W4 = diageye(5) + 2r4*r4' # positive definite matrix

        mxv_distribution = Wishart(5, W4)
        mxv_samples      = [ rand(rng, mxv_distribution) for _ in 1:20_000 ]
        mxv_sample_list  = SampleList(mxv_samples)

        @test isapprox(mean(mxv_sample_list), mean(mxv_distribution), atol = 1.0)
        @test isapprox(var(mxv_sample_list), var(mxv_distribution), atol = 3.0)
        @test isapprox(cov(mxv_sample_list), cov(mxv_distribution), atol = 10.0)
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

    @testset "SampleListMeta" begin 
        # @test_throws ErrorException get_meta(SampleList([ 0.1 ]))

        # rng = MersenneTwister(1234)

        # for uweights in [ rand(rng, 2), rand(rng, 2) ],
        #     entropy in [ -75.6, 234.2 ],
        #     logproposal in [ NormalMeanVariance(-3.0, 1.0), Gamma(2.0, 4.0), (x) -> x + 1.0 ],
        #     logintegrand in [ NormalMeanPrecision(-1.0, 2.0), Beta(3.0, 4.0), (x) -> log(abs(x)) ]

        #     meta = SampleListMeta(uweights, entropy, logproposal, logintegrand)
        #     sl   = SampleList([ 0.1, 0.1 ], [ 0.5, 0.5 ], meta)

        #     @test get_meta(sl) === meta
        #     @test get_unnormalised_weights(sl) == uweights
        #     @test get_entropy(sl) == entropy
        #     @test get_logproposal(sl) == logproposal
        #     @test get_logintegrand(sl) == logintegrand

        #     some_random_numbers = rand(rng, 100)

        #     __call(dist::Distribution, x) = logpdf(dist, x)
        #     __call(dist::Function, x)     = dist(x)

        #     @test map(e -> call_logproposal(sl, e), some_random_numbers) == map(e -> __call(logproposal, e), some_random_numbers)
        #     @test map(e -> call_logintegrand(sl, e), some_random_numbers) == map(e -> __call(logintegrand, e), some_random_numbers)
        # end

    end

end

end
