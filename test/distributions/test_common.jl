module DistributionsCommonTest

using Test
using ReactiveMP
using Distributions
using Random

@testset "Distributions Common" begin

    # Categorical comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality

    @testset "Bernoulli × Categorical" begin
        # These tests were using the `isapprox` on Categorical, but for some reason `isapprox` was broken for Categorical (upstream issue)
        # https://github.com/JuliaStats/Distributions.jl/issues/1675
        # We use probvec instead
        import ReactiveMP: probvec

        @test probvec(prod(ProdAnalytical(), Bernoulli(0.5), Categorical([0.5, 0.5]))) ≈ probvec(Categorical([0.5, 0.5]))
        @test probvec(prod(ProdAnalytical(), Bernoulli(0.1), Categorical(0.4, 0.6))) ≈ probvec(Categorical([1 - 0.14285714285714285, 0.14285714285714285]))
        @test probvec(prod(ProdAnalytical(), Bernoulli(0.78), Categorical([0.95, 0.05]))) ≈ probvec(Categorical([1 - 0.1572580645161291, 0.1572580645161291]))
        @test probvec(prod(ProdAnalytical(), Bernoulli(0.5), Categorical([0.3, 0.3, 0.4]))) ≈ probvec(Categorical([0.5, 0.5, 0]))
        @test probvec(prod(ProdAnalytical(), Bernoulli(0.5), Categorical([1.0]))) ≈ probvec(Categorical([1.0, 0]))
    end
end

end
