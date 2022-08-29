module NormalNaturalParametersTest

using Test
using ReactiveMP

@testset "NormalMeanPrecision" begin
    @testset "Constructor" begin
        @test standardDist(NormalNaturalParameters(1, -1)) ≈ NormalWeightedMeanPrecision(1, 2)
    end
end

end
