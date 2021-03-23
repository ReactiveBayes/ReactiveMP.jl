module WishartTest

using Test
using ReactiveMP
using Distributions
using Random
using LinearAlgebra

@testset "Wishart" begin

    # Wishart comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality

    @testset "vague" begin
        @test_throws MethodError vague(Wishart)

        d = vague(Wishart, 3)

        @test typeof(d) <: Wishart
        @test mean(d) == Matrix(Diagonal(3 * ReactiveMP.huge * ones(3)))
    end

    @testset "prod" begin
        v1 = [ 9.0 -3.4; -3.4 11.0 ]
        v2 = [ 10.2 -3.3; -3.3 5.0 ]
        v3 = [ 8.1 -2.7; -2.7 9.0 ]

        @test prod(ProdPreserveParametrisation(), Wishart(3, v1), Wishart(3, v2)) ≈ Wishart(3, [4.776325721474591 -1.6199382410125422; -1.6199382410125422 3.3487476649765537])
        @test prod(ProdPreserveParametrisation(), Wishart(4, v1), Wishart(4, v3)) ≈ Wishart(5, [4.261143738311623 -1.5064864332819319; -1.5064864332819319 4.949867121624725])
        @test prod(ProdPreserveParametrisation(), Wishart(5, v2), Wishart(4, v3)) ≈ Wishart(6, [4.51459128065395 -1.4750681198910067; -1.4750681198910067 3.129155313351499])

        @test prod(ProdBestSuitableParametrisation(), Wishart(3, v1), Wishart(3, v2)) ≈ Wishart(3, [4.776325721474591 -1.6199382410125422; -1.6199382410125422 3.3487476649765537])
        @test prod(ProdBestSuitableParametrisation(), Wishart(4, v1), Wishart(4, v3)) ≈ Wishart(5, [4.261143738311623 -1.5064864332819319; -1.5064864332819319 4.949867121624725])
        @test prod(ProdBestSuitableParametrisation(), Wishart(5, v2), Wishart(4, v3)) ≈ Wishart(6, [4.51459128065395 -1.4750681198910067; -1.4750681198910067 3.129155313351499])
    end

    @testset "ndims" begin
        @test ndims(vague(Wishart, 3)) === 3
        @test ndims(vague(Wishart, 4)) === 4
        @test ndims(vague(Wishart, 5)) === 5
    end

end

end
