module BetaTest

using Test
using ReactiveMP
using Distributions
using Random

@testset "Beta" begin

    # Beta comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality

    @testset "vague" begin
        d = vague(Beta)

        @test typeof(d) <: Beta
        @test mean(d) === 0.5
        @test params(d) === (1.0, 1.0)
    end

    @testset "prod" begin
        @test prod(ProdPreserveParametrisation(), Beta(3.0, 2.0), Beta(2.0, 1.0)) ≈ Beta(4.0, 2.0)
        @test prod(ProdPreserveParametrisation(), Beta(7.0, 1.0), Beta(0.1, 4.5)) ≈ Beta(6.1, 4.5)
        @test prod(ProdPreserveParametrisation(), Beta(1.0, 3.0), Beta(0.2, 0.4)) ≈ Beta(0.19999999999999996, 2.4)
        
        @test prod(ProdBestSuitableParametrisation(), Beta(3.0, 2.0), Beta(2.0, 1.0)) ≈ Beta(4.0, 2.0)
        @test prod(ProdBestSuitableParametrisation(), Beta(7.0, 1.0), Beta(0.1, 4.5)) ≈ Beta(6.1, 4.5)
        @test prod(ProdBestSuitableParametrisation(), Beta(1.0, 3.0), Beta(0.2, 0.4)) ≈ Beta(0.19999999999999996, 2.4)
    end

    @testset "logmean" begin
        @test logmean(Beta(1.0, 3.0)) ≈ -1.8333333333333335
        @test logmean(Beta(0.1, 0.3)) ≈ -7.862370395825961
        @test logmean(Beta(4.5, 0.3)) ≈ -0.07197681436958758
    end

    @testset "mirroredlogmean" begin
        @test mirroredlogmean(Beta(1.0, 3.0)) ≈ -0.33333333333333337
        @test mirroredlogmean(Beta(0.1, 0.3)) ≈ -0.9411396776150167
        @test mirroredlogmean(Beta(4.5, 0.3)) ≈ -4.963371962929249
    end

end

end
