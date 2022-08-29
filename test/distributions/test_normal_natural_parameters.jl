module NormalNaturalParametersTest

using Test
using ReactiveMP
using Distributions

@testset "NormalMeanPrecision" begin
    @testset "Constructor" begin
        @test standardDist(NormalNaturalParameters(1, -1)) ≈ NormalWeightedMeanPrecision(1, 2)
    end
    
    @testset "lognormalizer" begin
        @test lognormalizer(NormalNaturalParameters(1, -2)) ≈ (log(2) - 1/8)
    end

    @testset "logpdf" begin
        @test logpdf(NormalNaturalParameters(1, -1), 0) ≈ logpdf(NormalWeightedMeanPrecision(1, 2), 0)
    end
end

end
