module SampleListTest

using Test
using ReactiveMP
using Random
using Distributions
using LinearAlgebra
using StaticArrays
using StableRNGs
using FastCholesky

import ReactiveMP: deep_eltype, get_samples, get_weights, sample_list_zero_element
import ReactiveMP: get_meta, is_meta_present, get_unnormalised_weights, get_entropy, get_logproposal, get_logintegrand
import ReactiveMP: call_logproposal, call_logintegrand
import ReactiveMP: transform_samples, transform_weights!
import ReactiveMP: approximate_prod_with_sample_list
import ReactiveMP: xtlog, mirrorlog
import ReactiveMP: WishartMessage

@testset "SampleList" begin
    @testset "Internal functions" begin
        @test sample_list_zero_element(SampleList([1.0, 1.0])) === 0.0
        @test sample_list_zero_element(SampleList([[1.0, 1.0], [1.0, 1.0]])) == [0.0, 0.0]
        @test sample_list_zero_element(SampleList([[1.0 1.0; 1.0 1.0], [1.0 1.0; 1.0 1.0]])) == [0.0 0.0; 0.0 0.0]
        @test sample_list_zero_element(SampleList([[1.0 1.0 1.0; 1.0 1.0 1.0], [1.0 1.0 1.0; 1.0 1.0 1.0]])) == [0.0 0.0 0.0; 0.0 0.0 0.0]
        @test sample_list_zero_element(SampleList([[1.0; 1.0], [1.0; 1.0]])) == [0.0; 0.0]
    end

    @testset "Constructor" begin
        rng = StableRNG(1234)

        for N in [5, 10, 100], type in [Float64, Float32, BigFloat]
            scalar_samples    = rand(rng, type, N)
            scalar_samplelist = SampleList(scalar_samples)

            @test collect(get_samples(scalar_samplelist)) == first.(collect(scalar_samplelist))
            @test collect(get_weights(scalar_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(scalar_samplelist) === type
            @test eltype(scalar_samplelist) === Tuple{type, type}
            @test eltype(get_weights(scalar_samplelist)) === type
            @test variate_form(scalar_samplelist) === Univariate
            @test is_meta_present(scalar_samplelist) === false

            scalar_weights = rand(rng, type, N)

            @test_throws AssertionError SampleList(scalar_samples, scalar_weights)
            scalar_weights ./= sum(scalar_weights)
            scalar_samplelist = SampleList(scalar_samples, scalar_weights)

            @test collect(get_samples(scalar_samplelist)) == first.(collect(scalar_samplelist))
            @test collect(get_weights(scalar_samplelist)) == scalar_weights
            @test deep_eltype(scalar_samplelist) === type
            @test eltype(scalar_samplelist) === Tuple{type, type}
            @test eltype(get_weights(scalar_samplelist)) === type
            @test variate_form(scalar_samplelist) === Univariate
            @test is_meta_present(scalar_samplelist) === false

            vector_samples    = [rand(rng, type, 2) for _ in 1:N]
            vector_samplelist = SampleList(vector_samples)

            @test collect(get_samples(vector_samplelist)) == first.(collect(vector_samplelist))
            @test collect(get_weights(vector_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(vector_samplelist) === type
            @test eltype(vector_samplelist) === Tuple{SVector{2, type}, type}
            @test eltype(get_weights(vector_samplelist)) === type
            @test variate_form(vector_samplelist) === Multivariate
            @test is_meta_present(vector_samplelist) === false

            vector_weights = rand(rng, type, N)
            @test_throws AssertionError SampleList(vector_samples, vector_weights)
            vector_weights ./= sum(vector_weights)
            vector_samplelist = SampleList(vector_samples, vector_weights)

            @test collect(get_samples(vector_samplelist)) == first.(collect(vector_samplelist))
            @test collect(get_weights(vector_samplelist)) == vector_weights
            @test deep_eltype(vector_samplelist) === type
            @test eltype(vector_samplelist) === Tuple{SVector{2, type}, type}
            @test eltype(get_weights(vector_samplelist)) === type
            @test variate_form(vector_samplelist) === Multivariate
            @test is_meta_present(vector_samplelist) === false

            matrix_samples    = [rand(rng, type, 2, 2) for _ in 1:N]
            matrix_samplelist = SampleList(matrix_samples)

            @test collect(get_samples(matrix_samplelist)) == first.(collect(matrix_samplelist))
            @test collect(get_weights(matrix_samplelist)) == fill(one(type) / N, N)
            @test deep_eltype(matrix_samplelist) === type
            @test eltype(matrix_samplelist) === Tuple{SMatrix{2, 2, type, 4}, type}
            @test eltype(get_weights(matrix_samplelist)) === type
            @test variate_form(matrix_samplelist) === Matrixvariate
            @test is_meta_present(matrix_samplelist) === false

            matrix_weights = rand(rng, type, N)
            @test_throws AssertionError SampleList(matrix_samples, matrix_weights)
            matrix_weights ./= sum(matrix_weights)
            matrix_samplelist = SampleList(matrix_samples, matrix_weights)

            @test collect(get_samples(matrix_samplelist)) == first.(collect(matrix_samplelist))
            @test collect(get_weights(matrix_samplelist)) == matrix_weights
            @test deep_eltype(matrix_samplelist) === type
            @test eltype(matrix_samplelist) === Tuple{SMatrix{2, 2, type, 4}, type}
            @test eltype(get_weights(matrix_samplelist)) === type
            @test variate_form(matrix_samplelist) === Matrixvariate
            @test is_meta_present(matrix_samplelist) === false
        end

        @test_throws AssertionError SampleList(rand(10), rand(5))
        @test_throws AssertionError SampleList(rand(5), rand(10))
        @test_throws AssertionError SampleList(rand(5), [-1 for _ in 1:5])

        @test_throws AssertionError SampleList([rand(10) for _ in 1:10], rand(5))
        @test_throws AssertionError SampleList([rand(5) for _ in 1:5], rand(10))
        @test_throws AssertionError SampleList([rand(5) for _ in 1:5], [-1 for _ in 1:5])

        @test_throws AssertionError SampleList([rand(10, 10) for _ in 1:10], rand(5))
        @test_throws AssertionError SampleList([rand(5, 5) for _ in 1:5], rand(10))
        @test_throws AssertionError SampleList([rand(5, 5) for _ in 1:5], [-1 for _ in 1:5])
    end

    @testset "Statistics" begin
        rng = StableRNG(42)

        # All
        for N in [5, 10, 100]
            scalar_samples = rand(rng, N)
            scalar_weights = rand(rng, N)
            scalar_weights ./= sum(scalar_weights)
            scalar_samplelist = SampleList(scalar_samples, scalar_weights)
            arbitrary_f = (x) -> x .+ 1

            # Checking i = 1:2 that cache is not corrupted
            for i in 1:2
                @test mean(scalar_samplelist) ≈ sum(scalar_weights .* scalar_samples)
                @test mean(log, scalar_samplelist) ≈ sum(scalar_weights .* log.(scalar_samples))
                @test mean(xtlog, scalar_samplelist) ≈ sum(scalar_weights .* scalar_samples .* log.(scalar_samples))
                @test mean(mirrorlog, scalar_samplelist) ≈ sum(scalar_weights .* log.(1.0 .- scalar_samples))
                @test mean(arbitrary_f, scalar_samplelist) ≈ sum(scalar_weights .* arbitrary_f.(scalar_samples))
            end

            vector_samples = [rand(rng, 2) for _ in 1:N]
            vector_weights = rand(rng, N)
            vector_weights ./= sum(vector_weights)
            vector_samplelist = SampleList(vector_samples, vector_weights)

            # Checking i = 1:2 that cache is not corrupted
            for i in 1:2
                @test mean(vector_samplelist) ≈ sum(vector_weights .* vector_samples)
                @test mean(log, vector_samplelist) ≈ sum(vector_weights .* map(e -> log.(e), (vector_samples)))
                @test mean(xtlog, vector_samplelist) ≈ sum(vector_weights .* map(e -> e .* log.(e), (vector_samples)))
                @test mean(arbitrary_f, vector_samplelist) ≈ sum(vector_weights .* map(arbitrary_f, (vector_samples)))
            end

            matrix_samples = [rand(rng, 2, 2) for _ in 1:N]
            matrix_weights = rand(rng, N)
            matrix_weights ./= sum(matrix_weights)
            matrix_samplelist = SampleList(matrix_samples, matrix_weights)

            # Checking i = 1:2 that cache is not corrupted
            for i in 1:2
                @test mean(matrix_samplelist) ≈ sum(matrix_weights .* matrix_samples)
                @test mean(log, matrix_samplelist) ≈ sum(matrix_weights .* map(e -> log.(e), matrix_samples))
                @test mean(xtlog, matrix_samplelist) ≈ sum(matrix_weights .* map(e -> e .* log.(e), matrix_samples))
                @test mean(arbitrary_f, matrix_samplelist) ≈ sum(matrix_weights .* map(arbitrary_f, matrix_samples))
            end
        end

        uni_distribution = Gamma(rand(rng) + 1, rand(rng) + 2)
        uni_samples      = rand(rng, uni_distribution, 20_000)
        uni_sample_list  = SampleList(uni_samples)

        uni_distribution2 = NormalMeanVariance(rand(rng) + 1, rand(rng) + 2)
        uni_samples2      = rand(rng, uni_distribution2, 20_000)
        uni_sample_list2  = SampleList(uni_samples2)

        m               = rand(rng, 3)
        r               = rand(rng, 3)
        Σ               = I + 2r * r'
        mv_distribution = MvNormal(m, Σ)
        mv_samples      = [rand(rng, mv_distribution) for _ in 1:20_000]
        mv_sample_list  = SampleList(mv_samples)

        W1               = rand(rng, 3, 4)
        r2               = rand(rng, 3)
        W2               = I + 2r2 * r2'
        r3               = rand(rng, 4)
        W3               = I + 2r3 * r3'
        mxv_distribution = MatrixNormal(W1, W2, W3)
        mxv_samples      = [rand(rng, mxv_distribution) for _ in 1:20_000]
        mxv_sample_list  = SampleList(mxv_samples)

        # Checking i = 1:2 that cache is not corrupted
        for i in 1:2
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
        end

        mv_distribution = Dirichlet(rand(rng, 3))
        mv_samples      = [rand(rng, mv_distribution) for _ in 1:20_000]
        mv_sample_list  = SampleList(mv_samples)

        @test isapprox(mean(log, uni_sample_list), mean(log, uni_distribution); atol = 0.02)
        @test isapprox(mean(log, mv_sample_list), mean(log, mv_distribution); atol = 0.5)

        @test isapprox(mean(xtlog, uni_sample_list), mean(xtlog, uni_distribution); atol = 0.25)

        r4 = rand(rng, 5)
        W4 = I + 2r4 * r4'

        mxv_distribution = Wishart(5, W4)
        mxv_samples      = [rand(rng, mxv_distribution) for _ in 1:20_000]
        mxv_sample_list  = SampleList(mxv_samples)

        # Checking i = 1:2 that cache is not corrupted
        for i in 1:2
            @test isapprox(mean(mxv_sample_list), mean(mxv_distribution), atol = 1.0)
            @test isapprox(var(mxv_sample_list), var(mxv_distribution), atol = 2.5)
            @test isapprox(cov(mxv_sample_list), cov(mxv_distribution), atol = 5.0)
        end
    end

    @testset "vague" begin
        @test variate_form(vague(SampleList)) === Univariate
        @test variate_form(vague(SampleList, 2)) === Multivariate
        @test variate_form(vague(SampleList, 2, 2)) === Matrixvariate
        @test variate_form(vague(SampleList, (3, 4))) === Matrixvariate

        @test ndims(vague(SampleList)) === 1
        @test ndims(vague(SampleList, 2)) === 2
        @test ndims(vague(SampleList, 2, 2)) === (2, 2)
        @test ndims(vague(SampleList, (3, 4))) === (3, 4)

        for nsamples in [10, 100, 1000]
            @test variate_form(vague(SampleList; nsamples = nsamples)) === Univariate
            @test variate_form(vague(SampleList, 2; nsamples = nsamples)) === Multivariate
            @test variate_form(vague(SampleList, 2, 2; nsamples = nsamples)) === Matrixvariate
            @test variate_form(vague(SampleList, (3, 4); nsamples = nsamples)) === Matrixvariate

            @test length(vague(SampleList; nsamples = nsamples)) === nsamples
            @test length(vague(SampleList, 2; nsamples = nsamples)) === nsamples
            @test length(vague(SampleList, 2, 2; nsamples = nsamples)) === nsamples
            @test length(vague(SampleList, (3, 4); nsamples = nsamples)) === nsamples

            @test ndims(vague(SampleList; nsamples = nsamples)) === 1
            @test ndims(vague(SampleList, 2; nsamples = nsamples)) === 2
            @test ndims(vague(SampleList, 2, 2; nsamples = nsamples)) === (2, 2)
            @test ndims(vague(SampleList, (3, 4); nsamples = nsamples)) === (3, 4)

            @test size(vague(SampleList; nsamples = nsamples)) === (nsamples,)
            @test size(vague(SampleList, 2; nsamples = nsamples)) === (nsamples,)
            @test size(vague(SampleList, 2, 2; nsamples = nsamples)) === (nsamples,)
            @test size(vague(SampleList, (3, 4); nsamples = nsamples)) === (nsamples,)
        end
    end

    @testset "SampleListMeta" begin
        @test_throws ErrorException get_meta(SampleList([0.1]))

        rng = StableRNG(1234)

        for uweights in [rand(rng, 2), rand(rng, 2)],
            entropy in [-75.6, 234.2],
            logproposal in [NormalMeanVariance(-3.0, 1.0), Gamma(2.0, 4.0), (x) -> x + 1.0],
            logintegrand in [NormalMeanPrecision(-1.0, 2.0), Beta(3.0, 4.0), (x) -> log(abs(x))]

            meta = SampleListMeta(uweights, entropy, logproposal, logintegrand)
            sl   = SampleList([0.1, 0.1], [0.5, 0.5], meta)

            @test get_meta(sl) === meta
            @test get_unnormalised_weights(sl) == uweights
            @test get_entropy(sl) == entropy
            @test get_logproposal(sl) == logproposal
            @test get_logintegrand(sl) == logintegrand
            @test is_meta_present(sl) === true

            some_random_numbers = rand(rng, 100)

            __call(dist::Distribution, x) = logpdf(dist, x)
            __call(dist::Function, x)     = dist(x)

            @test map(e -> call_logproposal(sl, e), some_random_numbers) == map(e -> __call(logproposal, e), some_random_numbers)
            @test map(e -> call_logintegrand(sl, e), some_random_numbers) == map(e -> __call(logintegrand, e), some_random_numbers)
        end
    end

    @testset "Iteration utilities" begin
        rng = StableRNG(42)

        uni_distribution = Uniform(-10rand(rng), 10rand(rng))

        μ = rand(rng, 3)
        L1 = rand(rng, 3, 3)
        Σ = L1' * L1
        mv_distribution = MvNormal(μ, Σ)

        L2 = rand(rng, 3, 3)
        W = L2' * L2
        mvx_distribution = Wishart(3, W)

        # Entity to entity
        f1(e) = e .+ 1
        f2(e) = exp.(e)

        # Entity to Number
        f3(e::Number) = e + 1
        f3(e::AbstractVector) = norm(e .+ 1)
        f3(e::AbstractMatrix) = det(e .+ 1)

        # Entity to Vector
        f4(e::Number)         = @SVector [e, e]
        f4(e::AbstractVector) = reverse(e)
        f4(e::AbstractMatrix) = diag(e)

        # Entity to Matrix
        f5(e::Number)         = @SMatrix [e+1 e; e e+1]
        f5(e::AbstractVector) = SMatrix{length(e), length(e)}(Diagonal(ones(length(e))))
        f5(e::AbstractMatrix) = [e[1, 1]+1 e[1, 2]; e[2, 1] e[2, 2]+1]

        for N in (500, 1000, 5_000)
            for distribution in (uni_distribution, mv_distribution, mvx_distribution)
                samples = [rand(rng, distribution) for _ in 1:N]
                weights = ones(N) ./ N
                samplelist = SampleList(samples, weights)

                @test collect(samplelist) == collect(zip(samples, weights))
                @test map(i -> samplelist[i], 1:N) == collect(zip(samples, weights))

                for f in (f1, f2, f3, f4, f5)
                    @test all(map(e -> all(e[1] .≈ e[2]), zip(collect(transform_samples(f, samplelist)), collect(zip(map(f, samples), weights)))))
                    @test all(map(e -> all(e[1] .≈ e[2]), zip(map(i -> (f(samplelist[i][1]), samplelist[i][2]), 1:N), collect(zip(f.(samples), weights)))))
                end

                iter = N:-1:1
                index = 0

                old_weights = copy(weights)

                transform_weights!(w -> w * iter[index += 1], samplelist)

                newweights = map(prod, zip(old_weights, iter))
                newweights ./= sum(newweights)

                @test get_weights(samplelist) ≈ newweights
            end
        end
    end

    @testset "prod approximations" begin
        rng = StableRNG(1234)

        posdefm(rng, s) = begin
            L = rand(rng, s, s)
            L' * L
        end

        sizes  = [10_000, 15_000, 20_000]
        inputs = [(x = NormalMeanPrecision(3.0, 7.0), y = NormalMeanVariance(-4.0, 6.0), mean_tol = [1e-1, 1e-1, 1e-1], cov_tol = [1e-1, 1e-1, 1e-1], entropy_tol = [1e-1, 1e-1, 1e-1]), (x = NormalMeanVariance(3.0, 7.0), y = NormalWeightedMeanPrecision(4.0, 6.0), mean_tol = [1e-1, 1e-1, 1e-1], cov_tol = [1e-1, 1e-1, 1e-1], entropy_tol = [1e-1, 1e-1, 1e-1]), (x = GammaShapeRate(3.0, 7.0), y = GammaShapeScale(4.0, 6.0), mean_tol = [1e-1, 1e-1, 1e-1], cov_tol = [1e-1, 1e-1, 1e-1], entropy_tol = [3e-1, 3e-1, 3e-1]), (x = MvNormalMeanCovariance(10rand(rng, 4), posdefm(rng, 4)), y = MvNormalMeanPrecision(10rand(rng, 4), posdefm(rng, 4)), mean_tol = [3e-1, 3e-1, 3e-1], cov_tol = [6e-1, 6e-1, 6e-1], entropy_tol = [4e-1, 4e-1, 4e-1]), (x = WishartMessage(10.0, cholinv(posdefm(rng, 3))), y = WishartMessage(5.0, cholinv(posdefm(rng, 3))), mean_tol = [7e-1, 7e-1, 7e-1], cov_tol = [5e-1, 5e-1, 5e-1], entropy_tol = [2e-1, 2e-1, 2e-1])]

        for (i, N) in enumerate(sizes)
            for input in inputs
                analytical = prod(ProdAnalytical(), input[:x], input[:y])
                approximation = approximate_prod_with_sample_list(rng, input[:x], input[:y], N)

                @test is_meta_present(approximation) === true
                @test length(approximation) === N

                μᵣ, Σᵣ = mean_cov(analytical)
                μₐ, Σₐ = mean_cov(approximation)

                @test norm(μᵣ .- μₐ) < input[:mean_tol][i]
                @test norm(Σᵣ .- Σₐ) < input[:cov_tol][i]
                @test abs(entropy(analytical) - entropy(approximation)) < input[:entropy_tol][i]

                # Second order approximation here
                if (variate_form(input[:x]) === Univariate)
                    analytical2 = prod(ProdAnalytical(), analytical, input[:x])
                    approximation2 = approximate_prod_with_sample_list(rng, input[:x], approximation, N)

                    @test is_meta_present(approximation2) === true
                    @test length(approximation2) === N

                    μᵣ, Σᵣ = mean_cov(analytical2)
                    μₐ, Σₐ = mean_cov(approximation2)

                    @test norm(μᵣ .- μₐ) < input[:mean_tol][i]
                    @test norm(Σᵣ .- Σₐ) < input[:cov_tol][i]
                    @test abs(entropy(analytical2) - entropy(approximation2)) < input[:entropy_tol][i]
                end
            end
        end
    end
end

end
