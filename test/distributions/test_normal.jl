module NormalTest

using Test
using ReactiveMP
using Random
using LinearAlgebra
using Distributions

import ReactiveMP: convert_eltype

@testset "Normal" begin
    @testset "Univariate conversions" begin
        check_basic_statistics = (left, right) -> begin
            @test mean(left) ≈ mean(right)
            @test median(left) ≈ median(right)
            @test mode(left) ≈ mode(right)
            @test weightedmean(left) ≈ weightedmean(right)
            @test var(left) ≈ var(right)
            @test std(left) ≈ std(right)
            @test cov(left) ≈ cov(right)
            @test invcov(left) ≈ invcov(right)
            @test precision(left) ≈ precision(right)
            @test entropy(left) ≈ entropy(right)
            @test pdf(left, 1.0) ≈ pdf(right, 1.0)
            @test pdf(left, -1.0) ≈ pdf(right, -1.0)
            @test pdf(left, 0.0) ≈ pdf(right, 0.0)
            @test logpdf(left, 1.0) ≈ logpdf(right, 1.0)
            @test logpdf(left, -1.0) ≈ logpdf(right, -1.0)
            @test logpdf(left, 0.0) ≈ logpdf(right, 0.0)
        end

        types  = ReactiveMP.union_types(UnivariateNormalDistributionsFamily{Float64})
        etypes = ReactiveMP.union_types(UnivariateNormalDistributionsFamily)

        rng = MersenneTwister(1234)

        for type in types
            left = convert(type, rand(rng, Float64), rand(rng, Float64))
            for type in [types..., etypes...]
                right = convert(type, left)
                check_basic_statistics(left, right)

                p1 = prod(ProdPreserveTypeLeft(), left, right)
                @test typeof(p1) <: typeof(left)

                p2 = prod(ProdPreserveTypeRight(), left, right)
                @test typeof(p2) <: typeof(right)

                p3 = prod(ProdAnalytical(), left, right)

                check_basic_statistics(p1, p2)
                check_basic_statistics(p2, p3)
                check_basic_statistics(p1, p3)
            end
        end
    end

    @testset "Multivariate conversions" begin
        check_basic_statistics = (left, right, dims) -> begin
            @test mean(left) ≈ mean(right)
            @test mode(left) ≈ mode(right)
            @test weightedmean(left) ≈ weightedmean(right)
            @test var(left) ≈ var(right)
            @test cov(left) ≈ cov(right)
            @test invcov(left) ≈ invcov(right)
            @test logdetcov(left) ≈ logdetcov(right)
            @test precision(left) ≈ precision(right)
            @test length(left) === length(right)
            @test ndims(left) === ndims(right)
            @test size(left) === size(right)
            @test entropy(left) ≈ entropy(right)
            @test all(mean_cov(left) .≈ mean_cov(right))
            @test all(mean_invcov(left) .≈ mean_invcov(right))
            @test all(mean_precision(left) .≈ mean_precision(right))
            @test all(weightedmean_cov(left) .≈ weightedmean_cov(right))
            @test all(weightedmean_invcov(left) .≈ weightedmean_invcov(right))
            @test all(weightedmean_precision(left) .≈ weightedmean_precision(right))
            @test pdf(left, fill(1.0, dims)) ≈ pdf(right, fill(1.0, dims))
            @test pdf(left, fill(-1.0, dims)) ≈ pdf(right, fill(-1.0, dims))
            @test pdf(left, fill(0.0, dims)) ≈ pdf(right, fill(0.0, dims))
            @test logpdf(left, fill(1.0, dims)) ≈ logpdf(right, fill(1.0, dims))
            @test logpdf(left, fill(-1.0, dims)) ≈ logpdf(right, fill(-1.0, dims))
            @test logpdf(left, fill(0.0, dims)) ≈ logpdf(right, fill(0.0, dims))
        end

        types  = ReactiveMP.union_types(MultivariateNormalDistributionsFamily{Float64})
        etypes = ReactiveMP.union_types(MultivariateNormalDistributionsFamily)

        dims = (2, 3, 5)
        rng  = MersenneTwister(1234)

        for dim in dims
            for type in types
                left = convert(type, rand(rng, Float64, dim), Matrix(Diagonal(rand(rng, Float64, dim))))
                for type in [types..., etypes...]
                    right = convert(type, left)
                    check_basic_statistics(left, right, dim)

                    p1 = prod(ProdPreserveTypeLeft(), left, right)
                    @test typeof(p1) <: typeof(left)

                    p2 = prod(ProdPreserveTypeRight(), left, right)
                    @test typeof(p2) <: typeof(right)

                    p3 = prod(ProdAnalytical(), left, right)

                    check_basic_statistics(p1, p2, dim)
                    check_basic_statistics(p2, p3, dim)
                    check_basic_statistics(p1, p3, dim)
                end
            end
        end
    end

    @testset "JointNormal" begin

        # `@inferred` for type-stability check
        @test @inferred(mean_cov(convert(JointNormal, (1.0,), (1.0,))) == (1.0, 1.0))
        @test @inferred(mean_cov(convert(JointNormal, (1.0,), (2.0,))) == (1.0, 2.0))
        @test @inferred(ndims(convert(JointNormal, (1.0,), (2.0,)))) === 1

        @test getmarginal(convert(JointNormal, (1.0,), (1.0,)), 1) == NormalMeanVariance(1.0, 1.0)
        @test getmarginal(convert(JointNormal, (1.0,), (2.0,)), 1) == NormalMeanVariance(1.0, 2.0)

        @test @inferred(mean_cov(convert(JointNormal, ([1.0, 1.0],), ([1.0 0.0; 0.0 1.0],)))) == ([1.0, 1.0], [1.0 0.0; 0.0 1.0])
        @test @inferred(ndims(convert(JointNormal, ([1.0, 1.0],), ([1.0 0.0; 0.0 1.0],)))) === 2
        @test getmarginal(convert(JointNormal, ([1.0, 1.0],), ([1.0 0.0; 0.0 1.0],)), 1) == MvNormalMeanCovariance([1.0, 1.0], [1.0 0.0; 0.0 1.0])

        @test @inferred(mean_cov(convert(JointNormal, (1.0, -1.0), (1.0, 2.0)))) == ([1.0, -1.0], [1.0 0.0; 0.0 2.0])
        @test @inferred(mean_cov(convert(JointNormal, (3.0, 4.0), (2.0, 1.0)))) == ([3.0, 4.0], [2.0 0.0; 0.0 1.0])
        @test @inferred(ndims(convert(JointNormal, (3.0, 4.0), (2.0, 1.0)))) === 2

        @test getmarginal(convert(JointNormal, (1.0, -1.0), (1.0, 2.0)), 1) == NormalMeanVariance(1.0, 1.0)
        @test getmarginal(convert(JointNormal, (1.0, -1.0), (1.0, 2.0)), 2) == NormalMeanVariance(-1.0, 2.0)

        @test getmarginal(convert(JointNormal, (3.0, 4.0), (2.0, 1.0)), 1) == NormalMeanVariance(3.0, 2.0)
        @test getmarginal(convert(JointNormal, (3.0, 4.0), (2.0, 1.0)), 2) == NormalMeanVariance(4.0, 1.0)

        @test @inferred(mean_cov(convert(JointNormal, (1.0, [1.0, -1.0]), (1.0, [2.0 -1.0; -1.0 3.0])))) == ([1.0, 1.0, -1.0], [1.0 0.0 0.0; 0.0 2.0 -1.0; 0.0 -1.0 3.0])
        @test @inferred(mean_cov(convert(JointNormal, ([1.0, -1.0], [2.0, -2.0]), ([2.0 -1.0; -1.0 3.0], [3.0 -1.0; -1.0 2.0])))) ==
            ([1.0, -1.0, 2.0, -2.0], [2.0 -1.0 0.0 0.0; -1.0 3.0 0.0 0.0; 0.0 0.0 3.0 -1.0; 0.0 0.0 -1.0 2.0])

        @test @inferred(ndims(convert(JointNormal, ([1.0, -1.0], [2.0, -2.0]), ([2.0 -1.0; -1.0 3.0], [3.0 -1.0; -1.0 2.0])))) === 4

        @test getmarginal(convert(JointNormal, (1.0, [1.0, -1.0]), (1.0, [2.0 -1.0; -1.0 3.0])), 1) == NormalMeanVariance(1.0, 1.0)
        @test getmarginal(convert(JointNormal, (1.0, [1.0, -1.0]), (1.0, [2.0 -1.0; -1.0 3.0])), 2) == MvNormalMeanCovariance([1.0, -1.0], [2.0 -1.0; -1.0 3.0])

        @test @inferred(mean_cov(convert(JointNormal, MvNormalMeanCovariance([0.0], [2.0]), ((),)))) == (0.0, 2.0)
        @test @inferred(mean_cov(convert(JointNormal, MvNormalMeanCovariance([0.0, 1.0], [2.0 -0.5; -0.5 1.0]), ((2,),)))) == ([0.0, 1.0], [2.0 -0.5; -0.5 1.0])
        @test @inferred(mean_cov(convert(JointNormal, MvNormalMeanCovariance([0.0, 1.0], [2.0 -0.5; -0.5 1.0]), ((1,), (1,))))) == ([0.0, 1.0], [2.0 -0.5; -0.5 1.0])
        @test @inferred(mean_cov(convert(JointNormal, MvNormalMeanCovariance([0.0, 1.0], [2.0 -0.5; -0.5 1.0]), ((), ())))) == ([0.0, 1.0], [2.0 -0.5; -0.5 1.0])

        @test @inferred(ndims(convert(JointNormal, MvNormalMeanCovariance([0.0], [2.0]), ((),)))) === 1
        @test @inferred(ndims(convert(JointNormal, MvNormalMeanCovariance([0.0, 1.0], [2.0 -0.5; -0.5 1.0]), ((2,),)))) === 2
        @test @inferred(ndims(convert(JointNormal, MvNormalMeanCovariance([0.0, 1.0], [2.0 -0.5; -0.5 1.0]), ((1,), (1,))))) === 2
        @test @inferred(ndims(convert(JointNormal, MvNormalMeanCovariance([0.0, 1.0], [2.0 -0.5; -0.5 1.0]), ((), ())))) === 2

        @test getmarginal(convert(JointNormal, MvNormalMeanCovariance([0.0], [2.0]), ((),)), 1) == NormalMeanVariance(0.0, 2.0)
        @test getmarginal(convert(JointNormal, MvNormalMeanCovariance([0.0, 1.0], [2.0 -0.5; -0.5 1.0]), ((2,),)), 1) == MvNormalMeanCovariance([0.0, 1.0], [2.0 -0.5; -0.5 1.0])
        @test getmarginal(convert(JointNormal, MvNormalMeanCovariance([0.0, 1.0], [2.0 -0.5; -0.5 1.0]), ((1,), (1,))), 1) == MvNormalMeanCovariance([0.0], [2.0;;])
        @test getmarginal(convert(JointNormal, MvNormalMeanCovariance([0.0, 1.0], [2.0 -0.5; -0.5 1.0]), ((), ())), 1) == NormalMeanVariance(0.0, 2.0)

        @test @inferred(mean_cov(convert_eltype(JointNormal, Float32, JointNormal(NormalMeanVariance(0.0, 1.0), ((),))))) === (0.0f0, 1.0f0)
        @test @inferred(mean_cov(convert_eltype(JointNormal, Float32, JointNormal(MvNormalMeanCovariance([0.0], [1.0;;]), ((),))))) === (0.0f0, 1.0f0)
    end

    @testset "Variate forms promotions" begin
        @test promote_variate_type(Univariate, NormalMeanVariance) === NormalMeanVariance
        @test promote_variate_type(Univariate, NormalMeanPrecision) === NormalMeanPrecision
        @test promote_variate_type(Univariate, NormalWeightedMeanPrecision) === NormalWeightedMeanPrecision

        @test promote_variate_type(Multivariate, NormalMeanVariance) === MvNormalMeanCovariance
        @test promote_variate_type(Multivariate, NormalMeanPrecision) === MvNormalMeanPrecision
        @test promote_variate_type(Multivariate, NormalWeightedMeanPrecision) === MvNormalWeightedMeanPrecision

        @test promote_variate_type(Univariate, MvNormalMeanCovariance) === NormalMeanVariance
        @test promote_variate_type(Univariate, MvNormalMeanPrecision) === NormalMeanPrecision
        @test promote_variate_type(Univariate, MvNormalWeightedMeanPrecision) === NormalWeightedMeanPrecision

        @test promote_variate_type(Multivariate, MvNormalMeanCovariance) === MvNormalMeanCovariance
        @test promote_variate_type(Multivariate, MvNormalMeanPrecision) === MvNormalMeanPrecision
        @test promote_variate_type(Multivariate, MvNormalWeightedMeanPrecision) === MvNormalWeightedMeanPrecision
    end

    @testset "Sampling univariate" begin
        rng = MersenneTwister(1234)

        for T in (Float32, Float64)
            let # NormalMeanVariance
                μ, v = 10randn(rng), 10rand(rng)
                d    = convert(NormalMeanVariance{T}, μ, v)

                @test typeof(rand(d)) <: T

                samples = rand(rng, d, 5_000)

                @test isapprox(mean(samples), μ, atol = 0.5)
                @test isapprox(var(samples), v, atol = 0.5)
            end

            let # NormalMeanPrecision
                μ, w = 10randn(rng), 10rand(rng)
                d    = convert(NormalMeanPrecision{T}, μ, w)

                @test typeof(rand(d)) <: T

                samples = rand(rng, d, 5_000)

                @test isapprox(mean(samples), μ, atol = 0.5)
                @test isapprox(inv(var(samples)), w, atol = 0.5)
            end

            let # WeightedMeanPrecision
                wμ, w = 10randn(rng), 10rand(rng)
                d     = convert(NormalWeightedMeanPrecision{T}, wμ, w)

                @test typeof(rand(d)) <: T

                samples = rand(rng, d, 5_000)

                @test isapprox(inv(var(samples)) * mean(samples), wμ, atol = 0.5)
                @test isapprox(inv(var(samples)), w, atol = 0.5)
            end
        end
    end

    @testset "Sampling multivariate" begin
        rng = MersenneTwister(1234)

        for n in (2, 3), T in (Float64,), nsamples in (10_000,)
            let # MvNormalMeanCovariance
                μ = randn(rng, n)
                L = randn(rng, n, n)
                Σ = L * L'

                d = convert(MvNormalMeanCovariance{T}, μ, Σ)

                @test typeof(rand(d)) <: Vector{T}

                samples = SampleList(Val((n,)), rand(rng, d, nsamples), fill(1 / nsamples, nsamples))

                @test isapprox(mean(samples), mean(d), atol = n * 0.5)
                @test isapprox(cov(samples), cov(d), atol = n * 0.5)
            end

            let # MvNormalMeanPrecision
                μ = randn(rng, n)
                L = randn(rng, n, n)
                W = L * L'

                d = convert(MvNormalMeanPrecision{T}, μ, W)

                @test typeof(rand(d)) <: Vector{T}

                samples = SampleList(Val((n,)), rand(rng, d, nsamples), fill(T(1 / nsamples), nsamples))

                @test isapprox(mean(samples), mean(d), atol = n * 0.5)
                @test isapprox(cov(samples), cov(d), atol = n * 0.5)
            end

            let # MvNormalWeightedMeanPrecision
                ξ = randn(rng, n)
                L = randn(rng, n, n)
                W = L * L'

                d = convert(MvNormalWeightedMeanPrecision{T}, ξ, W)

                @test typeof(rand(d)) <: Vector{T}

                samples = SampleList(Val((n,)), rand(rng, d, nsamples), fill(T(1 / nsamples), nsamples))

                @test isapprox(mean(samples), mean(d), atol = n * 0.5)
                @test isapprox(cov(samples), cov(d), atol = n * 0.5)
            end
        end
    end

    @testset "UnivariateNormalNaturalParameters" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(Distribution, UnivariateNormalNaturalParameters(i, -i)) == NormalWeightedMeanPrecision(i, 2 * i)

                @test convert(UnivariateNormalNaturalParameters, i, -i) == UnivariateNormalNaturalParameters(i, -i)
                @test convert(UnivariateNormalNaturalParameters, [i, -i]) == UnivariateNormalNaturalParameters(i, -i)
                @test convert(UnivariateNormalNaturalParameters{Float64}, i, -i) == UnivariateNormalNaturalParameters(i, -i)
                @test convert(UnivariateNormalNaturalParameters{Float64}, [i, -i]) == UnivariateNormalNaturalParameters(i, -i)
            end
        end

        @testset "lognormalizer" begin
            @test lognormalizer(UnivariateNormalNaturalParameters(1, -2)) ≈ -(log(2) - 1 / 8)
        end

        @testset "logpdf" begin
            for i in 1:10
                @test logpdf(UnivariateNormalNaturalParameters(i, -i), 0) ≈ logpdf(NormalWeightedMeanPrecision(i, 2 * i), 0)
            end
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(UnivariateNormalNaturalParameters(i, -i)) === true
                @test isproper(UnivariateNormalNaturalParameters(i, i)) === false
            end
        end
    end

    @testset "MultivariateNormalNaturalParameters" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(Distribution, MultivariateNormalNaturalParameters([i, 0], [-i 0; 0 -i])) ≈ MvGaussianWeightedMeanPrecision([i, 0], [2*i 0; 0 2*i])

                @test convert(MultivariateNormalNaturalParameters, [i, 0], [-i 0; 0 -i]) == MultivariateNormalNaturalParameters([i, 0], [-i 0; 0 -i])
                @test convert(MultivariateNormalNaturalParameters, [i, 0, -i, 0, 0, -i]) == MultivariateNormalNaturalParameters([i, 0], [-i 0; 0 -i])
                @test convert(MultivariateNormalNaturalParameters{Float64}, [i, 0], [-i 0; 0 -i]) == MultivariateNormalNaturalParameters([i, 0], [-i 0; 0 -i])
                @test convert(MultivariateNormalNaturalParameters{Float64}, [i, 0, -i, 0, 0, -i]) == MultivariateNormalNaturalParameters([i, 0], [-i 0; 0 -i])

                @test as_naturalparams(MultivariateNormalNaturalParameters, [i, 0], [-i 0; 0 -i]) == MultivariateNormalNaturalParameters([i, 0], [-i 0; 0 -i])
                @test as_naturalparams(MultivariateNormalNaturalParameters, [i, 0, -i, 0, 0, -i]) == MultivariateNormalNaturalParameters([i, 0], [-i 0; 0 -i])
            end
        end

        @testset "logpdf" begin
            for i in 1:10
                mv_np = MultivariateNormalNaturalParameters([i, 0], [-i 0; 0 -i])
                distribution = MvGaussianWeightedMeanPrecision([i, 0.0], [2*i -0.0; -0.0 2*i])
                @test logpdf(distribution, [0.0, 0.0]) ≈ logpdf(mv_np, [0.0, 0.0])
                @test logpdf(distribution, [1.0, 0.0]) ≈ logpdf(mv_np, [1.0, 0.0])
                @test logpdf(distribution, [1.0, 1.0]) ≈ logpdf(mv_np, [1.0, 1.0])
            end
        end

        @testset "lognormalizer" begin
            mt = zeros(Float64, 1, 1) .- 2.0
            @test lognormalizer(MultivariateNormalNaturalParameters([1], mt)) ≈ -(log(2) - 1 / 8)
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(MultivariateNormalNaturalParameters([i, 0], [-i 0; 0 -i])) === true
                @test isproper(MultivariateNormalNaturalParameters([i, 0], [i 0; 0 i])) === false
            end
        end
    end
end

end
