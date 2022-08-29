module NormalNaturalParametersTest

using Test
using ReactiveMP

@testset "NormalMeanPrecision" begin
    @testset "Constructor" begin
        @test standardDist(NormalNaturalParameters(1, -1)) ≈ NormalWeightedMeanPrecision(1, 2)
    end
    
    @testset "lognormalizer" begin
        @test lognormalizer(NormalNaturalParameters(1, -2)) ≈ (log(2) - 1/8)
    end
end

end
