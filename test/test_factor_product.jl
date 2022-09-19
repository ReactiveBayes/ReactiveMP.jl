module FactorProdTest

using Test
using ReactiveMP
using Distributions
using Random

@testset "FactorProduct" begin

    # Bernoulli comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality

    @testset "getindex" begin
        product = FactorProduct((Normal(), Gamma(), Normal()))
        @test product[1] === Normal()
        @test product[2] === Gamma()
        @test product[3] === Normal()
    end

    @testset "entopy" begin
        product = FactorProduct((Normal(), Gamma(), Normal()))
        @test entropy(product) ≈ sum(map(entropy, (Normal(), Gamma(), Normal())))
    end
end

end
