module MixtureModelTest

using Test
using ReactiveMP
using Distributions
using Random
using SpecialFunctions
using ReactiveMP: mvbeta, logmvbeta

# test this testset with $ make test testset='distributions:mixture_model'
@testset "Mixture Model" begin

    @testset "creation with bernoulli prior" begin
        component1 = Normal(0.1, 0.3)
        component2 = Normal(2, 2.6)
        prior = Bernoulli(0.3)

        dist = MixtureModel([component1, component2], prior)

        @test typeof(dist.prior) <: Categorical
    end

    @testset "moments with 2 components" begin
        component1 = Normal(0.1, 0.3)
        component2 = Normal(2, 2.6)
        prior = Bernoulli(0.3)

        dist = MixtureModel([component1, component2], prior)

        @test mean(dist) ≈ 0.3 * 0.1 + 0.7 * 2
        @test var(dist)  ≈ 0.3 * (0.3^2 + 0.1^2) + 0.7 * (2.6^2 + 2^2) - mean(dist)^2
    end

    @testset "moments with 3 components" begin
        component1 = NormalMeanPrecision(0.3, 2)
        component2 = NormalMeanVariance(0.5, 1.0)
        component3 = NormalMeanPrecision(0.7, 2)
        prior = Categorical([0.1, 0.3, 0.6])

        dist = MixtureModel([component1, component2, component3], prior)

        @test mean(dist) ≈ 0.3 * 0.1 + 0.5 * 0.3 + 0.7 * 0.6
        @test var(dist)  ≈ 0.1 * (1/2 + 0.3^2) + 0.3 * (1.0 + 0.5^2) + 0.6 * (0.7^2 + 1/2) - mean(dist)^2
    end

    @testset "parameters - different distributions" begin
        component1 = Beta(0.1, 0.3)
        component2 = Normal(2, 2.6)
        prior = Bernoulli(0.3)

        dist = MixtureModel([component1, component2], prior)

        @test mean(dist) ≈ 0.3 * (0.1 / (0.1 + 0.3)) + 0.7 * 2
        @test var(dist)  ≈ 0.3 * ((0.1*0.3)/((0.1 + 0.3)^2*(0.1 + 0.3 +1)) + (0.1 / (0.1 + 0.3))^2) + 0.7 * (2.6^2 + 2^2) - mean(dist)^2
    end

    @testset "prod normal" begin
        component1 = NormalMeanVariance(3, 1)
        component2 = NormalMeanVariance(2, 4)
        prior = Bernoulli(0.3)
        dist = MixtureModel([component1, component2], prior)

        new_dist = prod(ProdAnalytical(), dist, component1)

        sf1 = 0.3 * sqrt(1/(2π*(1 + 1)))
        sf2 = 0.7 * sqrt(1/(2π*(1 + 4)))*exp(-(3-2)^2/5/2)
        p = sf1 / (sf1 + sf2)

        @test new_dist.components[1] == convert(NormalWeightedMeanPrecision, NormalMeanVariance(3, 1/2))
        @test new_dist.components[2] == NormalWeightedMeanPrecision(3 + 2/4, 1/4 + 1)
        @test new_dist.prior ≈ Categorical([p, 1-p])
    end

    @testset "prod beta" begin
        component1 = Beta(3, 1)
        component2 = Beta(2, 4)
        prior = Bernoulli(0.3)
        dist = MixtureModel([component1, component2], prior)

        new_dist = prod(ProdAnalytical(), dist, component1)

        sf1 = 0.3 * beta(5, 1) / (beta(3, 1) * beta(3, 1))
        sf2 = 0.7 * beta(4, 4) / (beta(3, 1) * beta(2, 4))
        p = sf1 / (sf1 + sf2)

        @test new_dist.components[1] == Beta(5, 1)
        @test new_dist.components[2] == Beta(4, 4)
        @test new_dist.prior ≈ Categorical([p, 1-p])
    end

    @testset "prod bernoulli" begin
        component1 = Bernoulli(0.6)
        component2 = Bernoulli(0.1)
        prior = Bernoulli(0.3)
        dist = MixtureModel([component1, component2], prior)

        new_dist = prod(ProdAnalytical(), dist, component1)

        sf1 = 0.3 * ((1 - 0.6)*(1 - 0.6) + 0.6*0.6)
        sf2 = 0.7 * ((1 - 0.6)*(1 - 0.1) + 0.6*0.1)
        p = sf1 / (sf1 + sf2)

        @test new_dist.components[1] == Bernoulli(0.6*0.6 / (0.6*0.6 + 0.4*0.4))
        @test new_dist.components[2] == Bernoulli(0.1*0.6 / (0.1*0.6 + 0.9*0.4))
        @test new_dist.prior ≈ Categorical([p, 1-p])
    end

    @testset "prod dirichlet" begin
        component1 = Dirichlet([6 ,1.0, 2.5])
        component2 = Dirichlet([5 ,2.0, 9.5])
        prior = Bernoulli(0.3)
        dist = MixtureModel([component1, component2], prior)

        new_dist = prod(ProdAnalytical(), dist, component1)

        sf1 = 0.3 * mvbeta([11, 1, 4]) / (mvbeta([6, 1, 2.5]) * mvbeta([6, 1, 2.5]))
        sf2 = 0.7 * mvbeta([10, 2, 11]) / (mvbeta([5, 2.0, 9.5]) * mvbeta([6, 1, 2.5]))
        p = sf1 / (sf1 + sf2)

        @test new_dist.components[1] == Dirichlet([11, 1, 4])
        @test new_dist.components[2] == Dirichlet([10, 2, 11])
        @test new_dist.prior ≈ Categorical([p, 1-p])
    end

    @testset "prod categorical" begin
        component1 = Categorical([0.1, 0.6, 0.3])
        component2 = Categorical([0.4, 0.4, 0.2])
        prior = Bernoulli(0.3)
        dist = MixtureModel([component1, component2], prior)

        new_dist = prod(ProdAnalytical(), dist, component1)

        sf1 = 0.3 * (0.1*0.1 + 0.6*0.6 + 0.3*0.3)
        sf2 = 0.7 * (0.1*0.4 + 0.6*0.4 + 0.3*0.2)
        p = sf1 / (sf1 + sf2)

        @test new_dist.components[1] == Categorical([0.1*0.1, 0.6*0.6, 0.3*0.3]/(0.1*0.1 + 0.6*0.6 + 0.3*0.3))
        @test new_dist.components[2] == Categorical([0.1*0.4, 0.6*0.4, 0.3*0.2]/(0.1*0.4 + 0.6*0.4 + 0.3*0.2))
        @test new_dist.prior ≈ Categorical([p, 1-p])
    end

    @testset "prod gamma" begin
        component1 = GammaShapeRate(3, 1)
        component2 = GammaShapeRate(2, 4)
        prior = Bernoulli(0.3)
        dist = MixtureModel([component1, component2], prior)

        new_dist = prod(ProdAnalytical(), dist, component1)

        sf1 = 0.3 * gamma(5) * 1^3 * 1^3 / gamma(3) / gamma(3) / 2^5
        sf2 = 0.7 * gamma(4) * 1^3 * 4^2 / gamma(3) / gamma(2) / 5^4
        p = sf1 / (sf1 + sf2)

        @test new_dist.components[1] == GammaShapeRate(5, 2)
        @test new_dist.components[2] == GammaShapeRate(4, 5)
        @test new_dist.prior ≈ Categorical([p, 1-p])
    end

end
end
