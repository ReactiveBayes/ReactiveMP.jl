module BernoulliTest

using Test
using ReactiveMP
using Distributions
using Random
using ReactiveMP: compute_logscale

@testset "Bernoulli" begin

    # Bernoulli comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ReactiveMP.jl specific functionality

    @testset "vague" begin
        d = vague(Bernoulli)

        @test typeof(d) <: Bernoulli
        @test mean(d) === 0.5
        @test succprob(d) === 0.5
        @test failprob(d) === 0.5
    end

    @testset "prod Bernoulli-Bernoulli" begin
        @test prod(ProdAnalytical(), Bernoulli(0.5), Bernoulli(0.5)) ≈ Bernoulli(0.5)
        @test prod(ProdAnalytical(), Bernoulli(0.1), Bernoulli(0.6)) ≈ Bernoulli(0.14285714285714285)
        @test prod(ProdAnalytical(), Bernoulli(0.78), Bernoulli(0.05)) ≈ Bernoulli(0.1572580645161291)
    end

    @testset "probvec" begin
        @test probvec(Bernoulli(0.5)) === (0.5, 0.5)
        @test probvec(Bernoulli(0.3)) === (0.7, 0.3)
        @test probvec(Bernoulli(0.6)) === (0.4, 0.6)
    end

    @testset "prod logscale Bernoulli-Bernoulli/Categorical" begin
        @test compute_logscale(Bernoulli(0.5), Bernoulli(0.5), Bernoulli(0.5)) ≈ log(0.5)
        @test compute_logscale(Bernoulli(1), Bernoulli(0.5), Bernoulli(1)) ≈ log(0.5)
        @test compute_logscale(Categorical([0.5, 0.5]), Bernoulli(0.5), Categorical([0.5, 0.5])) ≈ log(0.5)
        @test compute_logscale(Categorical([0.5, 0.5]), Categorical([0.5, 0.5]), Bernoulli(0.5)) ≈ log(0.5)
        @test compute_logscale(Categorical([1.0, 0.0]), Bernoulli(0.5), Categorical([1])) ≈ log(0.5)
        @test compute_logscale(Categorical([1.0, 0.0, 0.0]), Bernoulli(0.5), Categorical([1.0, 0, 0])) ≈ log(0.5)
    end

    @testset "BernoulliNaturalParameters" begin
        @test naturalparams(Bernoulli(0.5)) == BernoulliNaturalParameters(0.0)
        @test lognormalizer(naturalparams(Bernoulli(0.5))) ≈ -log(2)
        for i in 1:9
            bnp = naturalparams(Bernoulli(i / 10.0))
            @test convert(Distribution, bnp) ≈ Bernoulli(i / 10.0)
            @test logpdf(bnp, 1) ≈ logpdf(Bernoulli(i / 10.0), 1)
            @test logpdf(bnp, 0) ≈ logpdf(Bernoulli(i / 10.0), 0)

            @test convert(BernoulliNaturalParameters, i / 10.0) == BernoulliNaturalParameters(i / 10.0)
            @test convert(BernoulliNaturalParameters{Float64}, i / 10.0) == BernoulliNaturalParameters(i / 10.0)

            @test as_naturalparams(BernoulliNaturalParameters, i / 10.0) == BernoulliNaturalParameters(i / 10.0)
            @test as_naturalparams(BernoulliNaturalParameters{Float64}, i / 10.0) == BernoulliNaturalParameters(i / 10.0)
        end
        @test isproper(BernoulliNaturalParameters(10)) === true
    end
end

end
