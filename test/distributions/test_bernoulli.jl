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

    @testset "prod Bernoulli-Categorical" begin
        @test prod(ProdAnalytical(), Bernoulli(0.5), Categorical([1.0])) ≈ Categorical([1.0, 0.0])
        @test prod(ProdAnalytical(), Bernoulli(0.6), Categorical([0.7, 0.3])) ≈ Categorical([0.6086956521739131, 0.391304347826087])
        @test prod(ProdAnalytical(), Bernoulli(0.8), Categorical([0.2, 0.4, 0.4])) ≈ Categorical([0.1111111111111111, 0.8888888888888888, 0.0])
    end

    @testset "prod Categorical-Bernoulli" begin
        @test prod(ProdAnalytical(), Categorical([1.0]), Bernoulli(0.5)) ≈ Categorical([1.0, 0.0])
        @test prod(ProdAnalytical(), Categorical([0.7, 0.3]), Bernoulli(0.6)) ≈ Categorical([0.6086956521739131, 0.391304347826087])
        @test prod(ProdAnalytical(), Categorical([0.2, 0.4, 0.4]), Bernoulli(0.8)) ≈ Categorical([0.1111111111111111, 0.8888888888888888, 0.0])
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
        @test lognormalizer(naturalparams(Bernoulli(0.5))) ≈ log(2)
        b_99 = Bernoulli(0.99)
        for i in 1:9
            b = Bernoulli(i / 10.0)
            bnp = naturalparams(b)
            @test convert(Distribution, bnp) ≈ Bernoulli(i / 10.0)
            @test logpdf(bnp, 1) ≈ logpdf(Bernoulli(i / 10.0), 1)
            @test logpdf(bnp, 0) ≈ logpdf(Bernoulli(i / 10.0), 0)

            @test convert(BernoulliNaturalParameters, i / 10.0) == BernoulliNaturalParameters(i / 10.0)
            @test convert(BernoulliNaturalParameters{Float64}, i / 10.0) == BernoulliNaturalParameters(i / 10.0)

            @test as_naturalparams(BernoulliNaturalParameters, i / 10.0) == BernoulliNaturalParameters(i / 10.0)
            @test as_naturalparams(BernoulliNaturalParameters{Float64}, i / 10.0) == BernoulliNaturalParameters(i / 10.0)
            @test prod(ProdAnalytical(), convert(Distribution, naturalparams(b_99) - bnp), b) ≈ b_99
        end
        @test isproper(BernoulliNaturalParameters(10)) === true
    end
end

end
