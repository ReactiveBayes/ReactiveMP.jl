module DistributionsCommonTest

using Test
using ReactiveMP
using Distributions
using Random

@testset "Distributions Common" begin

    # Categorical comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality

    @testset "Bernoulli × Categorical" begin
        @test prod(ProdAnalytical(), Bernoulli(0.5), Categorical([0.5, 0.5])) ≈ Categorical([0.5, 0.5])
        @test prod(ProdAnalytical(), Bernoulli(0.1), Categorical(0.4, 0.6)) ≈ Categorical([1 - 0.14285714285714285, 0.14285714285714285])
        @test prod(ProdAnalytical(), Bernoulli(0.78), Categorical([0.95, 0.05])) ≈ Categorical([1 - 0.1572580645161291, 0.1572580645161291])
        @test prod(ProdAnalytical(), Bernoulli(0.5), Categorical([0.3, 0.3, 0.4])) ≈ Categorical([0.5, 0.5, 0])
        @test prod(ProdAnalytical(), Bernoulli(0.5), Categorical([1.0])) ≈ Categorical([1.0, 0])
    end
end

end
