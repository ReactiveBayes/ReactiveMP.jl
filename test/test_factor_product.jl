module FactorProdTest

using Test
using ReactiveMP
using Distributions
using Random

@testset "FactorProduct" begin

    # Bernoulli comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality
    
    
    multipliers = (Normal(), Gamma(), MvNormal([0,0], diageye(2)))
    @testset "getindex" begin
        product = FactorProduct(multipliers)
        @test product[1] === Normal()
        @test product[2] === Gamma()
        @test product[3] ≈ MvNormal([0,0], diageye(2))
    end

    @testset "entopy" begin
        product = FactorProduct(multipliers)
        @test entropy(product) ≈ mapreduce(entropy, +, multipliers)
    end
end

end
