module BernoulliTest

using Test
using ReactiveMP
using Distributions
using Random

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

    @testset "prod" begin
        @test prod(ProdAnalytical(), Bernoulli(0.5), Bernoulli(0.5)) ≈ Bernoulli(0.5)
        @test prod(ProdAnalytical(), Bernoulli(0.1), Bernoulli(0.6)) ≈ Bernoulli(0.14285714285714285)
        @test prod(ProdAnalytical(), Bernoulli(0.78), Bernoulli(0.05)) ≈ Bernoulli(0.1572580645161291)
    end

    @testset "probvec" begin
        @test probvec(Bernoulli(0.5)) === (0.5, 0.5)
        @test probvec(Bernoulli(0.3)) === (0.3, 0.7)
        @test probvec(Bernoulli(0.6)) === (0.6, 0.4)
    end

    @testset "naturalparameters" begin
        @test naturalparams(Bernoulli(0.5)).η === 0.0
        @test lognormalizer(naturalparams(Bernoulli(0.5))) ≈ -log(2)
        for i in 1:9
            bnp = naturalparams(Bernoulli(i / 10.0))
            @test standardDist(bnp) ≈ Bernoulli(i / 10.0)
            @test logpdf(bnp, 1) ≈ logpdf(Bernoulli(i / 10.0), 1)
            @test logpdf(bnp, 0) ≈ logpdf(Bernoulli(i / 10.0), 0)
        end
        @test isproper(BernoulliNaturalParameters(10)) === true
    end
end

end
