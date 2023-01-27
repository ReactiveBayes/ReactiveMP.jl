module ReactiveMPDistributionsTest

using Test
using ReactiveMP
using Distributions
using StaticArrays
using StableRNGs

import ReactiveMP: convert_eltype, deep_eltype, sampletype, samplefloattype, promote_sampletype, promote_samplefloattype
import ReactiveMP: FactorizedJoint

@testset "Distributions" begin
    function fixture_various_distributions(::Type{V} = Any; seed = abs(rand(Int)), Types = (Float32, Float64)) where {V}
        rng = StableRNG(seed)
        distributions = []

        # Add `Univariate` distributions
        for T in Types
            push!(distributions, PointMass(rand(rng, T)))
            push!(distributions, NormalMeanPrecision(rand(rng, T), rand(rng, T)))
            push!(distributions, NormalMeanVariance(rand(rng, T), rand(rng, T)))
            push!(distributions, NormalWeightedMeanPrecision(rand(rng, T), rand(rng, T)))
            push!(distributions, GammaShapeRate(rand(rng, T), rand(rng, T)))
            push!(distributions, GammaShapeScale(rand(rng, T), rand(rng, T)))
        end

        # Add `Multivariate` distributions
        for T in Types, n in 2:4
            push!(distributions, PointMass(rand(rng, T, n)))
            push!(distributions, MvNormalMeanPrecision(rand(rng, T, n)))
            push!(distributions, MvNormalMeanCovariance(rand(rng, T, n)))
            push!(distributions, MvNormalWeightedMeanPrecision(rand(rng, T, n)))
            push!(distributions, MvNormal(rand(rng, T, n)))
        end

        # Add `Matrixvariate` distributions
        for T in Types, n in 2:4
            push!(distributions, PointMass(rand(rng, T, n, n)))
            push!(distributions, Wishart(one(T), diageye(T, n)))
        end

        return filter((dist) -> variate_form(dist) <: V, distributions)
    end

    @testset "convert_eltype" begin
        for T in (Float32, Float64, BigFloat)
            @test @inferred(eltype(convert_eltype(T, [1.0, 1.0]))) === T
            @test @inferred(eltype(convert_eltype(T, [1.0 1.0; 1.0 1.0]))) === T
            @test @inferred(eltype(convert_eltype(T, 1.0))) === T
        end
    end

    @testset "sampletype" begin
        for distribution in fixture_various_distributions()
            sample = rand(distribution)
            @test @inferred(sampletype(distribution)) === typeof(sample)
        end
    end

    @testset "promote_sampletype" begin
        combinations = [
            Iterators.product(fixture_various_distributions(Univariate), fixture_various_distributions(Univariate)),
            Iterators.product(fixture_various_distributions(Multivariate), fixture_various_distributions(Multivariate)),
            Iterators.product(fixture_various_distributions(Matrixvariate), fixture_various_distributions(Matrixvariate))
        ]
        for combination in combinations
            for distributions in combination
                samples = rand.(distributions)
                @test @inferred(promote_sampletype(distributions...)) === promote_type(typeof.(samples)...)
            end
        end
    end

    @testset "samplefloattype" begin
        for distribution in fixture_various_distributions()
            sample = rand(distribution)
            @test @inferred(samplefloattype(distribution)) === deep_eltype(typeof(sample))
        end
    end

    @testset "promote_samplefloattype" begin
        combinations = [
            Iterators.product(fixture_various_distributions(Univariate), fixture_various_distributions(Univariate)),
            Iterators.product(fixture_various_distributions(Univariate), fixture_various_distributions(Matrixvariate)),
            Iterators.product(fixture_various_distributions(Multivariate), fixture_various_distributions(Multivariate)),
            Iterators.product(fixture_various_distributions(Multivariate), fixture_various_distributions(Matrixvariate)),
            Iterators.product(fixture_various_distributions(Matrixvariate), fixture_various_distributions(Matrixvariate)),
            Iterators.product(fixture_various_distributions(Univariate), fixture_various_distributions(Matrixvariate), fixture_various_distributions(Matrixvariate)),
        ]
        for combination in combinations
            for distributions in combination
                samples = rand.(distributions)
                @test @inferred(promote_samplefloattype(distributions...)) === promote_type(deep_eltype.(typeof.(samples))...)
            end
        end
    end

    @testset "FactorizedJoint" begin
        vmultipliers = [(NormalMeanPrecision(),), (NormalMeanVariance(), Beta(1.0, 1.0)), (Normal(), Gamma(), MvNormal(zeros(2), diageye(2)))]

        @testset "getindex" begin
            for multipliers in vmultipliers
                product = FactorizedJoint(multipliers)
                @test length(product) === length(multipliers)
                for i in length(multipliers)
                    @test product[i] === multipliers[i]
                end
            end
        end

        @testset "entropy" begin
            for multipliers in vmultipliers
                product = FactorizedJoint(multipliers)
                @test entropy(product) ≈ mapreduce(entropy, +, multipliers)
            end
        end

        @testset "isapprox" begin
            @test FactorizedJoint((NormalMeanVariance(),)) ≈ FactorizedJoint((NormalMeanVariance(),))
            @test !(FactorizedJoint((NormalMeanVariance(),)) ≈ FactorizedJoint((NormalMeanVariance(1, 1),)))

            @test FactorizedJoint((Gamma(1.0, 1.0), NormalMeanVariance(0.0, 1.0))) ≈ FactorizedJoint((Gamma(1.000001, 1.0), NormalMeanVariance(0.0, 1.0000000001))) atol = 1e-5
            @test !(FactorizedJoint((Gamma(1.0, 1.0), NormalMeanVariance(0.0, 1.0))) ≈ FactorizedJoint((Gamma(1.000001, 1.0), NormalMeanVariance(0.0, 5.0000000001))))
            @test !(FactorizedJoint((Gamma(1.0, 2.0), NormalMeanVariance(0.0, 1.0))) ≈ FactorizedJoint((Gamma(1.000001, 1.0), NormalMeanVariance(0.0, 1.0000000001))))
        end
    end
end

end
