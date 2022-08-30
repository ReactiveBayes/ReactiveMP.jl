module NormalNaturalParametersTest

using Test
using ReactiveMP
using Distributions

@testset "NormalNaturalParameters" begin
    @testset "Constructor" begin
        for i in 1:10
            @test standardDist(NormalNaturalParameters(i, -i)) ≈ NormalWeightedMeanPrecision(i, 2*i)
        end
    end

    @testset "lognormalizer" begin
        @test lognormalizer(NormalNaturalParameters(1, -2)) ≈ (log(2) - 1 / 8)
    end

    @testset "logpdf" begin
        for i in 1:10
            @test logpdf(NormalNaturalParameters(i, -i), 0) ≈ logpdf(NormalWeightedMeanPrecision(i, 2*i), 0)
        end
    end

    @testset "isproper" begin
        for i in 1:10
            @test isproper(NormalNaturalParameters(i, -i)) === true
            @test isproper(NormalNaturalParameters(i, i)) === false
        end
    end
end

end
