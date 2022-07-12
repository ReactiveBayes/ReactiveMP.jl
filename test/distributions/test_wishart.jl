module WishartTest

using Test
using ReactiveMP
using Distributions
using Random
using LinearAlgebra

@testset "Wishart" begin

    # Wishart comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality

    @testset "mean(:logdet)" begin
        @test mean(logdet, Wishart(3, [1.0 0.0; 0.0 1.0])) ≈ 0.845568670196936
        @test mean(
            logdet,
            Wishart(
                5,
                [
                    1.4659658963311604 1.111775094889733 0.8741034114800605
                    1.111775094889733 0.8746971141492232 0.6545661366809246
                    0.8741034114800605 0.6545661366809246 0.5498917856395482
                ]
            )
        ) ≈ -3.4633310802040693
    end

    @testset "vague" begin
        @test_throws MethodError vague(Wishart)

        d = vague(Wishart, 3)

        @test typeof(d) <: Wishart
        @test mean(d) == Matrix(Diagonal(3 * ReactiveMP.huge * ones(3)))
    end

    @testset "prod" begin
        v1 = [9.0 -3.4; -3.4 11.0]
        v2 = [10.2 -3.3; -3.3 5.0]
        v3 = [8.1 -2.7; -2.7 9.0]

        @test prod(ProdAnalytical(), WishartMessage(3, v1), WishartMessage(3, v2)) ≈
              WishartMessage(3, [4.776325721474591 -1.6199382410125422; -1.6199382410125422 3.3487476649765537])
        @test prod(ProdAnalytical(), WishartMessage(4, v1), WishartMessage(4, v3)) ≈
              WishartMessage(5, [4.261143738311623 -1.5064864332819319; -1.5064864332819319 4.949867121624725])
        @test prod(ProdAnalytical(), WishartMessage(5, v2), WishartMessage(4, v3)) ≈
              WishartMessage(6, [4.51459128065395 -1.4750681198910067; -1.4750681198910067 3.129155313351499])
    end

    @testset "ndims" begin
        @test ndims(vague(Wishart, 3)) === 3
        @test ndims(vague(Wishart, 4)) === 4
        @test ndims(vague(Wishart, 5)) === 5
    end
end

end
