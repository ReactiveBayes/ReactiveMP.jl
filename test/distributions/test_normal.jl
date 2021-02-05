module NormalTest

using Test
using ReactiveMP
using Random
using LinearAlgebra

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

        types = ReactiveMP.union_types(UnivariateNormalDistributionsFamily{Float64})
        rng   = MersenneTwister(1234)

        for type in types 
            left = convert(type, rand(rng, Float64), rand(rng, Float64))
            for type in types
                right = convert(type, left)
                check_basic_statistics(left, right)
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
            @test pdf(left, fill(1.0, dims))  ≈ pdf(right, fill(1.0, dims)) 
            @test pdf(left, fill(-1.0, dims)) ≈ pdf(right, fill(-1.0, dims))
            @test pdf(left, fill(0.0, dims))  ≈ pdf(right, fill(0.0, dims))
            @test logpdf(left, fill(1.0, dims))  ≈ logpdf(right, fill(1.0, dims)) 
            @test logpdf(left, fill(-1.0, dims)) ≈ logpdf(right, fill(-1.0, dims))
            @test logpdf(left, fill(0.0, dims))  ≈ logpdf(right, fill(0.0, dims))
        end

        types = ReactiveMP.union_types(MultivariateNormalDistributionsFamily{Float64})
        dims  = (2, 3, 5)
        rng   = MersenneTwister(1234)

        for dim in dims
            for type in types 
                left = convert(type, rand(rng, Float64, dim), Matrix(Diagonal(rand(rng, Float64, dim))))
                for type in types
                    right = convert(type, left)
                    check_basic_statistics(left, right, dim)
                end
            end
        end

    end

end

end