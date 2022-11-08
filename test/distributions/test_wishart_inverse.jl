module InverseWishartTest

using Test
using ReactiveMP
using Distributions
using Random
using LinearAlgebra
using StableRNGs

import ReactiveMP: InverseWishartMessage

@testset "InverseWishartMessage" begin
    @testset "common" begin
        @test InverseWishartMessage <: Distribution
        @test InverseWishartMessage <: ContinuousDistribution
        @test InverseWishartMessage <: MatrixDistribution

        @test value_support(InverseWishartMessage) === Continuous
        @test variate_form(InverseWishartMessage) === Matrixvariate
    end

    @testset "statistics" begin
        rng = StableRNG(42)
        # ν > dim(d) + 1
        for ν in 4:10
            L = randn(rng, ν - 2, ν - 2)
            S = L * L'
            d = InverseWishartMessage(ν, S)

            @test mean(d) == mean(InverseWishart(params(d)...))
            @test mode(d) == mode(InverseWishart(params(d)...))
        end

        # ν > dim(d) + 3
        for ν in 5:10
            L = randn(rng, ν - 4, ν - 4)
            S = L * L'
            d = InverseWishartMessage(ν, S)

            @test cov(d) == cov(InverseWishart(params(d)...))
            @test var(d) == var(InverseWishart(params(d)...))
        end
    end

    @testset "vague" begin
        @test_throws MethodError vague(InverseWishartMessage)

        dims = 3
        d1 = vague(InverseWishart, dims)

        @test typeof(d1) <: InverseWishart
        ν1, S1 = params(d1)
        @test ν1 == dims + 2
        @test S1 == tiny .* diageye(dims)

        @test mean(d1) == S1

        dims = 4
        d2 = vague(InverseWishart, dims)

        @test typeof(d2) <: InverseWishart
        ν2, S2 = params(d2)
        @test ν2 == dims + 2
        @test S2 == tiny .* diageye(dims)

        @test mean(d2) == S2
    end

    @testset "entropy" begin
        @test entropy(InverseWishartMessage(2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712])) ≈ 10.111427477184794
        @test entropy(InverseWishartMessage(5.0, diageye(4))) ≈ 8.939145914882221
    end

    @testset "convert" begin
        rng = StableRNG(42)
        for ν in 2:10
            L = randn(rng, ν, ν)
            S = L * L'
            d = InverseWishartMessage(ν, S)
            @test convert(InverseWishart, d) == InverseWishart(ν, S)
        end
    end

    @testset "mean(::typeof(logdet))" begin
        rng = StableRNG(123)
        ν, S = 2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(logdet, InverseWishartMessage(ν, S)), mean(logdet.(samples)), atol = 1e-2)

        ν, S = 4.0, diageye(3)
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(logdet, InverseWishartMessage(ν, S)), mean(logdet.(samples)), atol = 1e-2)
    end

    @testset "mean(::typeof(inv))" begin
        rng = StableRNG(321)
        ν, S = 2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(inv, InverseWishartMessage(ν, S)), mean(inv.(samples)), atol = 1e-2)
        @test isapprox(mean(cholinv, InverseWishartMessage(ν, S)), mean(cholinv.(samples)), atol = 1e-2)

        ν, S = 4.0, diageye(3)
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(inv, InverseWishartMessage(ν, S)), mean(inv.(samples)), atol = 1e-2)
        @test isapprox(mean(cholinv, InverseWishartMessage(ν, S)), mean(cholinv.(samples)), atol = 1e-2)
    end

    @testset "prod" begin
        d1 = InverseWishartMessage(3.0, diageye(2))
        d2 = InverseWishartMessage(-3.0, [0.6423504672769315 0.9203141654948761; 0.9203141654948761 1.528137747462735])

        @test prod(ProdAnalytical(), d1, d2) ≈ InverseWishartMessage(3.0, [1.6423504672769313 0.9203141654948761; 0.9203141654948761 2.528137747462735])

        d1 = InverseWishartMessage(4.0, diageye(3))
        d2 = InverseWishartMessage(-2.0, diageye(3))

        @test prod(ProdAnalytical(), d1, d2) ≈ InverseWishartMessage(6.0, 2 * diageye(3))
    end
end

end
