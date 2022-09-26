module NormalNaturalParametersTest

using Test
using ReactiveMP
using Distributions

@testset "UnivariateNormalNaturalParameters" begin
    @testset "Constructor" begin
        for i in 1:10
            @test convert(Distribution, UnivariateNormalNaturalParameters(i, -i)) ≈ NormalWeightedMeanPrecision(i, 2 * i)
        end
    end

    @testset "lognormalizer" begin
        @test lognormalizer(UnivariateNormalNaturalParameters(1, -2)) ≈ (log(2) - 1 / 8)
    end

    @testset "logpdf" begin
        for i in 1:10
            @test logpdf(UnivariateNormalNaturalParameters(i, -i), 0) ≈ logpdf(NormalWeightedMeanPrecision(i, 2 * i), 0)
        end
    end

    @testset "isproper" begin
        for i in 1:10
            @test isproper(UnivariateNormalNaturalParameters(i, -i)) === true
            @test isproper(UnivariateNormalNaturalParameters(i, i)) === false
        end
    end
end

end
