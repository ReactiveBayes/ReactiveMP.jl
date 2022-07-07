module MatrixDirichletTest

using Test
using ReactiveMP
using Distributions
using Random
using LinearAlgebra

@testset "InvWishart" begin
    @testset "common" begin
        @test InvWishart <: Distribution
        @test InvWishart <: ContinuousDistribution
        @test InvWishart <: MatrixDistribution

        @test value_support(InvWishart) === Continuous
        @test variate_form(InvWishart) === Matrixvariate
    end

    @testset "statistics" begin
        # ν > dim(d) + 1
        for ν in 4:10
            L = randn(ν - 2, ν - 2)
            S = L * L'
            d = InvWishart(ν, S)

            @test mean(d) == mean(InverseWishart(params(d)...))
            @test mode(d) == mode(InverseWishart(params(d)...))
        end

        # ν > dim(d) + 3
        for ν in 5:10
            L = randn(ν - 4, ν - 4)
            S = L * L'
            d = InvWishart(ν, S)

            @test cov(d) == cov(InverseWishart(params(d)...))
            @test var(d) == var(InverseWishart(params(d)...))
        end
    end

    @testset "vague" begin
        @test_throws MethodError vague(InvWishart)

        dims = 3
        d1 = vague(InvWishart, dims)

        @test typeof(d1) <: InvWishart
        @test d1.ν == dims
        @test d1.S == tiny .* diageye(dims)

        dims = 4
        d2 = vague(InvWishart, dims)

        @test typeof(d2) <: InvWishart
        @test d2.ν == dims
        @test d2.S == tiny .* diageye(dims)
    end

    @testset "entropy" begin
        @test entropy(
            InvWishart(2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712])
        ) ≈ 10.111427477184794
        @test entropy(InvWishart(5.0, diageye(4))) ≈ 8.939145914882221
    end

    @testset "convert" begin
        for ν in 2:10
            L = randn(ν, ν)
            S = L * L'
            d = InvWishart(ν, S)
            @test convert(InverseWishart, d) == InverseWishart(ν, S)
        end
    end

    @testset "mean(::typeof(logdet))" begin
        ν, S = 2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
        samples = rand(InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(logdet, InvWishart(ν, S)), mean(logdet.(samples)), atol=1e-2)

        ν, S = 4.0, diageye(3)
        samples = rand(InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(logdet, InvWishart(ν, S)), mean(logdet.(samples)), atol=1e-2)
    end

    @testset "mean(::typeof(inv))" begin
        ν, S = 2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
        samples = rand(InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(inv, InvWishart(ν, S)), mean(inv.(samples)), atol=1e-2)

        ν, S = 4.0, diageye(3)
        samples = rand(InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(inv, InvWishart(ν, S)), mean(inv.(samples)), atol=1e-2)
    end

    @testset "prod" begin
        d1 = InvWishart(3.0, diageye(2))
        d2 = InvWishart(-3.0, [0.6423504672769315 0.9203141654948761; 0.9203141654948761 1.528137747462735])

        @test prod(ProdAnalytical(), d1, d2) ≈
              InvWishart(3.0, [1.6423504672769313 0.9203141654948761; 0.9203141654948761 2.528137747462735])

        d1 = InvWishart(4.0, diageye(3))
        d2 = InvWishart(-2.0, diageye(3))

        @test prod(ProdAnalytical(), d1, d2) ≈
              InvWishart(6.0, 2 * diageye(3))
    end
end

end
