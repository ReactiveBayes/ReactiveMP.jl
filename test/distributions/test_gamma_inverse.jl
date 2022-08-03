module GammaInverseTest

using Test
#using Distributions
using ReactiveMP
using Random

@testset "GammaInverse" begin
    @testset "vague" begin
        d = vague(GammaInverse)
        @test typeof(d) <: GammaInverse
        @test mean(d) === 0.5
        @test params(d) === (1.0, 1.0)
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), GammaInverse(3.0, 2.0), GammaInverse(2.0, 1.0)) ≈ GammaInverse(4.0, 2.0)
        @test prod(ProdAnalytical(), GammaInverse(7.0, 1.0), GammaInverse(0.1, 4.5)) ≈ GammaInverse(6.1, 4.5)
        @test prod(ProdAnalytical(), GammaInverse(1.0, 3.0), GammaInverse(0.2, 0.4)) ≈ GammaInverse(0.19999999999999996, 2.4)
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, GammaInverse(1.0, 3.0)) ≈ -1.8333333333333335
        @test mean(log, GammaInverse(0.1, 0.3)) ≈ -7.862370395825961
        @test mean(log, GammaInverse(4.5, 0.3)) ≈ -0.07197681436958758
    end

    @testset "mean(::typeof(inv))" begin
        @test mean(inv, GammaInverse(1.0, 3.0)) ≈ -0.33333333333333337
        @test mean(inv, GammaInverse(0.1, 0.3)) ≈ -0.9411396776150167
        @test mean(inv, GammaInverse(4.5, 0.3)) ≈ -4.963371962929249
    end
end
end
