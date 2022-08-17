module GammaInverseTest

using Test
using ReactiveMP
using Distributions # params
using Random

# test this testset with $ make test testset='distributions:gamma_inverse'
@testset "GammaInverse" begin
    @testset "vague" begin
        d = vague(GammaInverse)
        @test typeof(d) <: GammaInverse
        @test mean(d) == huge
        @test params(d) == (2.0, huge)
    end

    # (α, θ) = (α_L + α_R + 1, θ_L + θ_R)
    @testset "prod" begin
        @test prod(ProdAnalytical(), GammaInverse(3.0, 2.0), GammaInverse(2.0, 1.0)) ≈ GammaInverse(6.0, 3.0)
        @test prod(ProdAnalytical(), GammaInverse(7.0, 1.0), GammaInverse(0.1, 4.5)) ≈ GammaInverse(8.1, 5.5)
        @test prod(ProdAnalytical(), GammaInverse(1.0, 3.0), GammaInverse(0.2, 0.4)) ≈ GammaInverse(2.2, 3.4)
    end

    # log(θ) - digamma(α)
    @testset "mean(::typeof(log))" begin
        @test mean(log, GammaInverse(1.0, 3.0)) ≈ 1.6758279535696414
        @test mean(log, GammaInverse(0.1, 0.3)) ≈ 9.21978213608514
        @test mean(log, GammaInverse(4.5, 0.3)) ≈ -2.5928437306854653
        @test mean(log, GammaInverse(42.0, 42.0)) ≈ 0.011952000346086233
    end

    # α / θ
    @testset "mean(::typeof(inv))" begin
        @test mean(inv, GammaInverse(1.0, 3.0)) ≈ 0.33333333333333333
        @test mean(inv, GammaInverse(0.1, 0.3)) ≈ 0.33333333333333337
        @test mean(inv, GammaInverse(4.5, 0.3)) ≈ 15.0000000000000000
        @test mean(inv, GammaInverse(42.0, 42.0)) ≈ 1.0000000000000000
    end
end
end
