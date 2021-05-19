module NormalTest

using Test
using ReactiveMP
using Random
using LinearAlgebra
using Distributions

@testset "Normal" begin
    
    @testset "Univariate conversions" begin

        check_basic_statistics = (left, right) -> begin
            @test mean(left)         ≈ mean(right)
            @test median(left)       ≈ median(right) 
            @test mode(left)         ≈ mode(right) 
            @test weightedmean(left) ≈ weightedmean(right)
            @test var(left)          ≈ var(right)
            @test std(left)          ≈ std(right)
            @test cov(left)          ≈ cov(right) 
            @test invcov(left)       ≈ invcov(right)
            @test precision(left)    ≈ precision(right)
            @test entropy(left)      ≈ entropy(right)
            @test pdf(left, 1.0)     ≈ pdf(right, 1.0) 
            @test pdf(left, -1.0)    ≈ pdf(right, -1.0)
            @test pdf(left, 0.0)     ≈ pdf(right, 0.0)
            @test logpdf(left, 1.0)  ≈ logpdf(right, 1.0)
            @test logpdf(left, -1.0) ≈ logpdf(right, -1.0)
            @test logpdf(left, 0.0)  ≈ logpdf(right, 0.0)
        end

        types  = ReactiveMP.union_types(UnivariateNormalDistributionsFamily{Float64})
        etypes = ReactiveMP.union_types(UnivariateNormalDistributionsFamily)

        rng   = MersenneTwister(1234)

        for type in types 
            left = convert(type, rand(rng, Float64), rand(rng, Float64))
            for type in [ types..., etypes... ]
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
            @test mean(left)         ≈ mean(right)
            @test mode(left)         ≈ mode(right) 
            @test weightedmean(left) ≈ weightedmean(right)
            @test var(left)          ≈ var(right)
            @test cov(left)          ≈ cov(right) 
            @test invcov(left)       ≈ invcov(right)
            @test logdetcov(left)    ≈ logdetcov(right)
            @test precision(left)    ≈ precision(right)
            @test length(left)       === length(right)
            @test ndims(left)        === ndims(right)
            @test size(left)         === size(right)
            @test entropy(left)      ≈ entropy(right)
            @test all(mean_cov(left) .≈ mean_cov(right))
            @test all(mean_invcov(left) .≈ mean_invcov(right))
            @test all(mean_precision(left) .≈ mean_precision(right))
            @test all(weightedmean_cov(left) .≈ weightedmean_cov(right))
            @test all(weightedmean_invcov(left) .≈ weightedmean_invcov(right))
            @test all(weightedmean_precision(left) .≈ weightedmean_precision(right))
            @test pdf(left, fill(1.0, dims))  ≈ pdf(right, fill(1.0, dims)) 
            @test pdf(left, fill(-1.0, dims)) ≈ pdf(right, fill(-1.0, dims))
            @test pdf(left, fill(0.0, dims))  ≈ pdf(right, fill(0.0, dims))
            @test logpdf(left, fill(1.0, dims))  ≈ logpdf(right, fill(1.0, dims)) 
            @test logpdf(left, fill(-1.0, dims)) ≈ logpdf(right, fill(-1.0, dims))
            @test logpdf(left, fill(0.0, dims))  ≈ logpdf(right, fill(0.0, dims))
        end

        types  = ReactiveMP.union_types(MultivariateNormalDistributionsFamily{Float64})
        etypes = ReactiveMP.union_types(MultivariateNormalDistributionsFamily)

        dims  = (2, 3, 5)
        rng   = MersenneTwister(1234)

        for dim in dims
            for type in types 
                left = convert(type, rand(rng, Float64, dim), Matrix(Diagonal(rand(rng, Float64, dim))))
                for type in [ types..., etypes... ]
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

    @testset "Variate forms promotions" begin
        
        @test promote_variate_type(Univariate, NormalMeanVariance)          === NormalMeanVariance
        @test promote_variate_type(Univariate, NormalMeanPrecision)         === NormalMeanPrecision
        @test promote_variate_type(Univariate, NormalWeightedMeanPrecision) === NormalWeightedMeanPrecision

        @test promote_variate_type(Multivariate, NormalMeanVariance)          === MvNormalMeanCovariance
        @test promote_variate_type(Multivariate, NormalMeanPrecision)         === MvNormalMeanPrecision
        @test promote_variate_type(Multivariate, NormalWeightedMeanPrecision) === MvNormalWeightedMeanPrecision

        @test promote_variate_type(Univariate, MvNormalMeanCovariance)        === NormalMeanVariance
        @test promote_variate_type(Univariate, MvNormalMeanPrecision)         === NormalMeanPrecision
        @test promote_variate_type(Univariate, MvNormalWeightedMeanPrecision) === NormalWeightedMeanPrecision

        @test promote_variate_type(Multivariate, MvNormalMeanCovariance)        === MvNormalMeanCovariance
        @test promote_variate_type(Multivariate, MvNormalMeanPrecision)         === MvNormalMeanPrecision
        @test promote_variate_type(Multivariate, MvNormalWeightedMeanPrecision) === MvNormalWeightedMeanPrecision

    end

end

end