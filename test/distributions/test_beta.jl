module BetaTest

using Test
using ReactiveMP
using Distributions
using Random

import ReactiveMP: mirrorlog
import SpecialFunctions: loggamma

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
        @test prod(ProdAnalytical(), Beta(3.0, 2.0), Beta(2.0, 1.0)) ≈ Beta(4.0, 2.0)
        @test prod(ProdAnalytical(), Beta(7.0, 1.0), Beta(0.1, 4.5)) ≈ Beta(6.1, 4.5)
        @test prod(ProdAnalytical(), Beta(1.0, 3.0), Beta(0.2, 0.4)) ≈ Beta(0.19999999999999996, 2.4)
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, Beta(1.0, 3.0)) ≈ -1.8333333333333335
        @test mean(log, Beta(0.1, 0.3)) ≈ -7.862370395825961
        @test mean(log, Beta(4.5, 0.3)) ≈ -0.07197681436958758
    end

    @testset "mean(::typeof(mirrorlog))" begin
        @test mean(mirrorlog, Beta(1.0, 3.0)) ≈ -0.33333333333333337
        @test mean(mirrorlog, Beta(0.1, 0.3)) ≈ -0.9411396776150167
        @test mean(mirrorlog, Beta(4.5, 0.3)) ≈ -4.963371962929249
    end

    @testset "BetaNaturalParameters" begin
        @testset "Constructor" begin
            for i in 0:10, j in 0:10
                @test convert(Distribution, BetaNaturalParameters(i, j)) == Beta(i+1, j+1)

                @test convert(BetaNaturalParameters, i, j) == BetaNaturalParameters(i, j)
                @test convert(BetaNaturalParameters, [i, j]) == BetaNaturalParameters(i, j)
            end
        end

        @testset "lognormalizer" begin
            @test lognormalizer(BetaNaturalParameters(0, 0)) ≈ 0
            @test lognormalizer(BetaNaturalParameters(1, 1)) ≈ -loggamma(4)
        end

        @testset "logpdf" begin
            for i in 0:10, j in 0:10
                @test logpdf(BetaNaturalParameters(i, j), 0) ≈ logpdf(Beta(i+1, j+1), 0)
                @test logpdf(BetaNaturalParameters(i, j), 0.5) ≈ logpdf(Beta(i+1, j+1), 0.5)
            end
        end

        @testset "isproper" begin
            for i in 0:10
                @test isproper(BetaNaturalParameters(i, i)) === true
            end
            for i in 1:10
                @test isproper(BetaNaturalParameters(-i, -i)) === false
            end
        end
    end
end

end
