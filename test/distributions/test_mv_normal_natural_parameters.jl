module MvNormalNaturalParametersTest

using Test
using ReactiveMP
using Distributions

@testset "MvNormalNaturalParameters" begin
    @testset "Constructor" begin
        for i in 1:10
            @test standardDist(MvNormalNaturalParameters([i, 0], [-i 0; 0 -i])) ≈
                  MvGaussianWeightedMeanPrecision([i, 0], [2*i 0; 0 2*i])
        end
    end

    @testset "logpdf" begin
        for i in 1:10
            mv_np = MvNormalNaturalParameters([i, 0], [-i 0; 0 -i])
            distribution = MvGaussianWeightedMeanPrecision([i, 0.0], [2*i -0.0; -0.0 2*i])
            @test logpdf(distribution, [0.0, 0.0]) ≈ logpdf(mv_np, [0.0, 0.0])
            @test logpdf(distribution, [1.0, 0.0]) ≈ logpdf(mv_np, [1.0, 0.0])
            @test logpdf(distribution, [1.0, 1.0]) ≈ logpdf(mv_np, [1.0, 1.0])
        end
    end

    @testset "lognormalizer" begin
        mt = zeros(Float64, 1, 1) .- 2.0
        @test lognormalizer(MvNormalNaturalParameters([1], mt)) ≈ (log(2) - 1 / 8)
    end

    @testset "isproper" begin
        for i in 1:10
            @test isproper(MvNormalNaturalParameters([i, 0], [-i 0; 0 -i])) === true
            @test isproper(MvNormalNaturalParameters([i, 0], [i 0; 0 i])) === false
        end
    end
end

end
