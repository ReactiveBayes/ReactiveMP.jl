module DistributionsCommonTest

using Test
using ReactiveMP
using Distributions
using Random

@testset "Distributions Common" begin

    # Categorical comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality

    @testset "Bernoulli × Categorical" begin
        @test prod(ProdPreserveParametrisation(), Bernoulli(0.3), Categorical([ 1/2, 1/2 ])) ≈ prod(ProdPreserveParametrisation(), Bernoulli(0.3), Bernoulli(0.5))
        @test prod(ProdPreserveParametrisation(), Bernoulli(0.1), Categorical([ 0.2, 0.8 ])) ≈ prod(ProdPreserveParametrisation(), Bernoulli(0.1), Bernoulli(0.2))
        @test prod(ProdPreserveParametrisation(), Bernoulli(0.9), Categorical([ 0.8, 0.2 ])) ≈ prod(ProdPreserveParametrisation(), Bernoulli(0.9), Bernoulli(0.8))
    end

end

end
