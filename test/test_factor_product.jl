module FactorProdTest

using Test
using ReactiveMP
using Distributions
using Random

function isapprox_test(x::FactorProduct, y::FactorProduct, output::Bool; atol::Real = 0)
    @test isapprox(x, y, atol = atol) === output
end

@testset "FactorProduct" begin

    # Bernoulli comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality

    multipliers = (Normal(), Gamma(), MvNormal([0, 0], diageye(2)))
    @testset "getindex" begin
        product = FactorProduct(multipliers)
        @test product[1] === Normal()
        @test product[2] === Gamma()
        @test product[3] ≈ MvNormal([0, 0], diageye(2))
    end

    @testset "entopy" begin
        product = FactorProduct(multipliers)
        @test entropy(product) ≈ mapreduce(entropy, +, multipliers)
    end

    @testset "isapprox" begin
        @test FactorProduct((NormalMeanVariance(),)) ≈ FactorProduct((NormalMeanVariance(),))
        @test !(FactorProduct((NormalMeanVariance(),)) ≈ FactorProduct((NormalMeanVariance(1, 1),)))
        isapprox_test(
            FactorProduct((vague(Gamma), NormalMeanVariance())),
            FactorProduct((Gamma(1.000001, 1.0e12), NormalMeanVariance())),
            true,
            atol = 0.00001
        )
        isapprox_test(
            FactorProduct((vague(Gamma), NormalMeanVariance())),
            FactorProduct((Gamma(1.000001, 1.0e12), NormalMeanVariance())),
            false
        )
    end
end

end
