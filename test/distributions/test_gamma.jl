module GammaTest

using Test
using ReactiveMP
using Random
using Distributions

import SpecialFunctions: loggamma
import ReactiveMP: xtlog

@testset "Gamma" begin
    @testset "Constructor" begin
        @test GammaShapeRate <: GammaDistributionsFamily
        @test GammaShapeScale <: GammaDistributionsFamily

        @test GammaShapeRate() == GammaShapeRate{Float64}(1.0, 1.0)
        @test GammaShapeRate(1.0) == GammaShapeRate{Float64}(1.0, 1.0)
        @test GammaShapeRate(1.0, 2.0) == GammaShapeRate{Float64}(1.0, 2.0)
        @test GammaShapeRate(1) == GammaShapeRate{Float64}(1.0, 1.0)
        @test GammaShapeRate(1, 2) == GammaShapeRate{Float64}(1.0, 2.0)
        @test GammaShapeRate(1.0, 2) == GammaShapeRate{Float64}(1.0, 2.0)
        @test GammaShapeRate(1, 2.0) == GammaShapeRate{Float64}(1.0, 2.0)
        @test GammaShapeRate(1.0f0) == GammaShapeRate{Float32}(1.0f0, 1.0f0)
        @test GammaShapeRate(1.0f0, 2.0f0) == GammaShapeRate{Float32}(1.0f0, 2.0f0)
        @test GammaShapeRate(1.0f0, 2) == GammaShapeRate{Float32}(1.0f0, 2.0f0)
        @test GammaShapeRate(1.0f0, 2.0) == GammaShapeRate{Float64}(1.0, 2.0)

        @test GammaShapeScale() == GammaShapeScale{Float64}(1.0, 1.0)
        @test GammaShapeScale(1.0) == GammaShapeScale{Float64}(1.0, 1.0)
        @test GammaShapeScale(1.0, 2.0) == GammaShapeScale{Float64}(1.0, 2.0)
        @test GammaShapeScale(1) == GammaShapeScale{Float64}(1.0, 1.0)
        @test GammaShapeScale(1, 2) == GammaShapeScale{Float64}(1.0, 2.0)
        @test GammaShapeScale(1.0, 2) == GammaShapeScale{Float64}(1.0, 2.0)
        @test GammaShapeScale(1, 2.0) == GammaShapeScale{Float64}(1.0, 2.0)
        @test GammaShapeScale(1.0f0) == GammaShapeScale{Float32}(1.0f0, 1.0f0)
        @test GammaShapeScale(1.0f0, 2.0f0) == GammaShapeScale{Float32}(1.0f0, 2.0f0)
        @test GammaShapeScale(1.0f0, 2) == GammaShapeScale{Float32}(1.0f0, 2.0f0)
        @test GammaShapeScale(1.0f0, 2.0) == GammaShapeScale{Float64}(1.0, 2.0)

        @test eltype(GammaShapeRate(1.0, 2.0)) === Float64
        @test eltype(GammaShapeRate(1.0f0, 2.0f0)) === Float32
    end

    @testset "vague" begin
        vague(GammaShapeScale) == Gamma(1.0, ReactiveMP.huge)
        vague(GammaShapeRate) == Gamma(1.0, ReactiveMP.tiny)
    end

    @testset "Stats methods for GammaShapeScale" begin
        dist1 = GammaShapeScale(1.0, 1.0)

        @test mean(dist1) === 1.0
        @test var(dist1) === 1.0
        @test cov(dist1) === 1.0
        @test shape(dist1) === 1.0
        @test scale(dist1) === 1.0
        @test rate(dist1) === 1.0
        @test entropy(dist1) ≈ 1.0

        dist2 = GammaShapeScale(1.0, 2.0)

        @test mean(dist2) === 2.0
        @test var(dist2) === 4.0
        @test cov(dist2) === 4.0
        @test shape(dist2) === 1.0
        @test scale(dist2) === 2.0
        @test rate(dist2) === inv(2.0)
        @test entropy(dist2) ≈ 1.6931471805599454

        dist3 = GammaShapeScale(2.0, 2.0)

        @test mean(dist3) === 4.0
        @test var(dist3) === 8.0
        @test cov(dist3) === 8.0
        @test shape(dist3) === 2.0
        @test scale(dist3) === 2.0
        @test rate(dist3) === inv(2.0)
        @test entropy(dist3) ≈ 2.2703628454614764

        dist4 = GammaShapeScale(257.37489915581654, 1/3.0)

        @test mean(dist4) ≈ 257.37489915581654/3.0
        @test var(dist4) ≈ 257.37489915581654/9.0
        @test shape(dist4) ≈ 257.37489915581654
        @test scale(dist4) ≈ inv(3.0)
        @test rate(dist4) ≈ 3.0
        @test entropy(dist4) ≈ 3.0942967450433203
        @test pdf(dist4, 86.2027941354432) ≈ 0.07400338986722949
        @test logpdf(dist4, 86.2027941354432) ≈ -2.6036443778105536    
    end

    @testset "Stats methods for GammaShapeRate" begin
        dist1 = GammaShapeRate(1.0, 1.0)

        @test mean(dist1) === 1.0
        @test var(dist1) === 1.0
        @test cov(dist1) === 1.0
        @test shape(dist1) === 1.0
        @test scale(dist1) === 1.0
        @test rate(dist1) === 1.0
        @test entropy(dist1) ≈ 1.0
        @test pdf(dist1, 1.0) ≈ 0.36787944117144233
        @test logpdf(dist1, 1.0) ≈ -1.0

        dist2 = GammaShapeRate(1.0, 2.0)

        @test mean(dist2) === inv(2.0)
        @test var(dist2) === inv(4.0)
        @test cov(dist2) === inv(4.0)
        @test shape(dist2) === 1.0
        @test scale(dist2) === inv(2.0)
        @test rate(dist2) === 2.0
        @test entropy(dist2) ≈ 0.3068528194400547
        @test pdf(dist2, 1.0) ≈ 0.2706705664732254
        @test logpdf(dist2, 1.0) ≈ -1.3068528194400546

        dist3 = GammaShapeRate(2.0, 2.0)

        @test mean(dist3) === 1.0
        @test var(dist3) === inv(2.0)
        @test cov(dist3) === inv(2.0)
        @test shape(dist3) === 2.0
        @test scale(dist3) === inv(2.0)
        @test rate(dist3) === 2.0
        @test entropy(dist3) ≈ 0.8840684843415857
        @test pdf(dist3, 1.0) ≈ 0.5413411329464508
        @test logpdf(dist3, 1.0) ≈ -0.6137056388801094

        dist4 = GammaShapeRate(257.37489915581654, 3.0)

        @test mean(dist4) ≈ 257.37489915581654/3.0
        @test var(dist4) ≈ 257.37489915581654/9.0
        @test shape(dist4) ≈ 257.37489915581654
        @test scale(dist4) ≈ inv(3.0)
        @test rate(dist4) ≈ 3.0
        @test entropy(dist4) ≈ 3.0942967450433203
        @test pdf(dist4, 86.2027941354432) ≈ 0.07400338986722949
        @test logpdf(dist4, 86.2027941354432) ≈ -2.6036443778105536        
    end

    @testset "GammaShapeRateNaturalParameters" begin
        for i in 2:10
            @test convert(Distribution, GammaNaturalParameters(i, -i)) ≈ GammaShapeRate(i + 1, i)
            @test Distributions.logpdf(GammaNaturalParameters(i, -i), 10) ≈ Distributions.logpdf(GammaShapeRate(i + 1, i), 10)
            @test isproper(GammaNaturalParameters(i, -i)) === true
            @test isproper(GammaNaturalParameters(-i, i)) === false

            @test convert(GammaNaturalParameters, i, -i) == GammaNaturalParameters(i, -i)
            @test convert(GammaNaturalParameters{Float64}, i, -i) == GammaNaturalParameters(i, -i)

            @test as_naturalparams(GammaNaturalParameters, i, -i) == GammaNaturalParameters(i, -i)
            @test as_naturalparams(GammaNaturalParameters{Float64}, i, -i) == GammaNaturalParameters(i, -i)
        end
    end

    @testset "Base methods" begin
        @test convert(GammaShapeScale{Float32}, GammaShapeScale()) == GammaShapeScale{Float32}(1.0f0, 1.0f0)
        @test convert(GammaShapeScale{Float64}, GammaShapeScale(1.0, 10.0)) == GammaShapeScale{Float64}(1.0, 10.0)
        @test convert(GammaShapeScale{Float64}, GammaShapeScale(1.0, 0.1)) == GammaShapeScale{Float64}(1.0, 0.1)
        @test convert(GammaShapeScale{Float64}, 1, 1) == GammaShapeScale{Float64}(1.0, 1.0)
        @test convert(GammaShapeScale{Float64}, 1, 10) == GammaShapeScale{Float64}(1.0, 10.0)
        @test convert(GammaShapeScale{Float64}, 1.0, 0.1) == GammaShapeScale{Float64}(1.0, 0.1)

        @test convert(GammaShapeRate{Float32}, GammaShapeRate()) == GammaShapeRate{Float32}(1.0f0, 1.0f0)
        @test convert(GammaShapeRate{Float64}, GammaShapeRate(1.0, 10.0)) == GammaShapeRate{Float64}(1.0, 10.0)
        @test convert(GammaShapeRate{Float64}, GammaShapeRate(1.0, 0.1)) == GammaShapeRate{Float64}(1.0, 0.1)
        @test convert(GammaShapeRate{Float64}, 1, 1) == GammaShapeRate{Float64}(1.0, 1.0)
        @test convert(GammaShapeRate{Float64}, 1, 10) == GammaShapeRate{Float64}(1.0, 10.0)
        @test convert(GammaShapeRate{Float64}, 1.0, 0.1) == GammaShapeRate{Float64}(1.0, 0.1)

        @test convert(GammaShapeRate, GammaShapeRate(2.0, 2.0)) == GammaShapeRate{Float64}(2.0, 2.0)
        @test convert(GammaShapeScale, GammaShapeRate(2.0, 2.0)) == GammaShapeScale{Float64}(2.0, 1.0 / 2.0)

        @test convert(GammaShapeRate, GammaShapeScale(2.0, 2.0)) == GammaShapeRate{Float64}(2.0, 1.0 / 2.0)
        @test convert(GammaShapeScale, GammaShapeScale(2.0, 2.0)) == GammaShapeScale{Float64}(2.0, 2.0)

        check_basic_statistics = (left, right) -> begin
            @test mean(left) ≈ mean(right)
            @test var(left) ≈ var(right)
            @test cov(left) ≈ cov(right)
            @test shape(left) ≈ shape(right)
            @test scale(left) ≈ scale(right)
            @test rate(left) ≈ rate(right)
            @test entropy(left) ≈ entropy(right)
            @test pdf(left, 1.0) ≈ pdf(right, 1.0)
            @test pdf(left, 10.0) ≈ pdf(right, 10.0)
            @test logpdf(left, 1.0) ≈ logpdf(right, 1.0)
            @test logpdf(left, 10.0) ≈ logpdf(right, 10.0)
            @test mean(log, left) ≈ mean(log, right)
            @test mean(loggamma, left) ≈ mean(loggamma, right)
            @test mean(xtlog, left) ≈ mean(xtlog, right)
        end

        types = ReactiveMP.union_types(GammaDistributionsFamily{Float64})
        rng   = MersenneTwister(1234)

        for type in types
            left = convert(type, rand(rng, Float64), rand(rng, Float64))
            for type in types
                right = convert(type, left)
                check_basic_statistics(left, right)
            end
        end
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), GammaShapeScale(1, 1), GammaShapeScale(1, 1)) == GammaShapeScale(1, 1 / 2)
        @test prod(ProdAnalytical(), GammaShapeScale(1, 2), GammaShapeScale(1, 1)) == GammaShapeScale(1, 2 / 3)
        @test prod(ProdAnalytical(), GammaShapeScale(1, 2), GammaShapeScale(1, 2)) == GammaShapeScale(1, 1)
        @test prod(ProdAnalytical(), GammaShapeScale(2, 2), GammaShapeScale(1, 2)) == GammaShapeScale(2, 1)
        @test prod(ProdAnalytical(), GammaShapeScale(2, 2), GammaShapeScale(2, 2)) == GammaShapeScale(3, 1)

        @test prod(ProdAnalytical(), GammaShapeRate(1, 1), GammaShapeRate(1, 1)) == GammaShapeRate(1, 2)
        @test prod(ProdAnalytical(), GammaShapeRate(1, 2), GammaShapeRate(1, 1)) == GammaShapeRate(1, 3)
        @test prod(ProdAnalytical(), GammaShapeRate(1, 2), GammaShapeRate(1, 2)) == GammaShapeRate(1, 4)
        @test prod(ProdAnalytical(), GammaShapeRate(2, 2), GammaShapeRate(1, 2)) == GammaShapeRate(2, 4)
        @test prod(ProdAnalytical(), GammaShapeRate(2, 2), GammaShapeRate(2, 2)) == GammaShapeRate(3, 4)

        @test prod(ProdAnalytical(), GammaShapeScale(1, 1), GammaShapeRate(1, 1)) == GammaShapeScale(1, 1 / 2)
        @test prod(ProdAnalytical(), GammaShapeScale(1, 2), GammaShapeRate(1, 1)) == GammaShapeScale(1, 2 / 3)
        @test prod(ProdAnalytical(), GammaShapeScale(1, 2), GammaShapeRate(1, 2)) == GammaShapeScale(1, 2 / 5)
        @test prod(ProdAnalytical(), GammaShapeScale(2, 2), GammaShapeRate(1, 2)) == GammaShapeScale(2, 2 / 5)
        @test prod(ProdAnalytical(), GammaShapeScale(2, 2), GammaShapeRate(2, 2)) == GammaShapeScale(3, 2 / 5)

        @test prod(ProdAnalytical(), GammaShapeRate(1, 1), GammaShapeScale(1, 1)) == GammaShapeRate(1, 2)
        @test prod(ProdAnalytical(), GammaShapeRate(1, 2), GammaShapeScale(1, 1)) == GammaShapeRate(1, 3)
        @test prod(ProdAnalytical(), GammaShapeRate(1, 2), GammaShapeScale(1, 2)) == GammaShapeRate(1, 5 / 2)
        @test prod(ProdAnalytical(), GammaShapeRate(2, 2), GammaShapeScale(1, 2)) == GammaShapeRate(2, 5 / 2)
        @test prod(ProdAnalytical(), GammaShapeRate(2, 2), GammaShapeScale(2, 2)) == GammaShapeRate(3, 5 / 2)
    end
end

end
