module NormalMeanVarianceTest

using Test
using ReactiveMP

@testset "NormalMeanVariance" begin

    @testset "Constructor" begin
        @test NormalMeanVariance()         == NormalMeanVariance{Float64}(0.0, 1.0)
        @test NormalMeanVariance(1.0)      == NormalMeanVariance{Float64}(1.0, 1.0)
        @test NormalMeanVariance(1.0, 2.0) == NormalMeanVariance{Float64}(1.0, 2.0)
        @test NormalMeanVariance(1)        == NormalMeanVariance{Float64}(1.0, 1.0)
        @test NormalMeanVariance(1, 2)     == NormalMeanVariance{Float64}(1.0, 2.0)
        @test NormalMeanVariance(1f0)      == NormalMeanVariance{Float32}(1f0, 1f0)
        @test NormalMeanVariance(1f0, 2f0) == NormalMeanVariance{Float32}(1f0, 2f0)

        @test eltype(NormalMeanVariance())         === Float64
        @test eltype(NormalMeanVariance(0.0))      === Float64
        @test eltype(NormalMeanVariance(0.0, 1.0)) === Float64
        @test eltype(NormalMeanVariance(0))        === Float64
        @test eltype(NormalMeanVariance(0, 1))     === Float64
        @test eltype(NormalMeanVariance(0f0))      === Float32
        @test eltype(NormalMeanVariance(0f0, 1f0)) === Float32
    end

    @testset "Stats methods" begin
        
        @test mean(NormalMeanVariance(0.0, 1.0))      === 0.0
        @test median(NormalMeanVariance(0.0, 1.0))    === 0.0
        @test mode(NormalMeanVariance(0.0, 1.0))      === 0.0
        @test var(NormalMeanVariance(0.0, 1.0))       === 1.0
        @test std(NormalMeanVariance(0.0, 1.0))       === 1.0
        @test cov(NormalMeanVariance(0.0, 1.0))       === 1.0
        @test invcov(NormalMeanVariance(0.0, 1.0))    === 1.0
        @test precision(NormalMeanVariance(0.0, 1.0)) === 1.0
        @test entropy(NormalMeanVariance(0.0, 1.0))   ≈ 1.41893853320467
        @test pdf(NormalMeanVariance(0.0, 1.0), 1.0)  ≈ 0.24197072451914337
        @test pdf(NormalMeanVariance(0.0, 1.0), -1.0) ≈ 0.24197072451914337
        @test pdf(NormalMeanVariance(0.0, 1.0), 0.0)  ≈ 0.3989422804014327
        @test logpdf(NormalMeanVariance(0.0, 1.0), 1.0)  ≈ -1.4189385332046727
        @test logpdf(NormalMeanVariance(0.0, 1.0), -1.0) ≈ -1.4189385332046727
        @test logpdf(NormalMeanVariance(0.0, 1.0), 0.0)  ≈ -0.9189385332046728

        @test mean(NormalMeanVariance(1.0, 1.0))      === 1.0
        @test median(NormalMeanVariance(1.0, 1.0))    === 1.0
        @test mode(NormalMeanVariance(1.0, 1.0))      === 1.0
        @test var(NormalMeanVariance(1.0, 1.0))       === 1.0
        @test std(NormalMeanVariance(1.0, 1.0))       === 1.0
        @test cov(NormalMeanVariance(1.0, 1.0))       === 1.0
        @test invcov(NormalMeanVariance(1.0, 1.0))    === 1.0
        @test precision(NormalMeanVariance(1.0, 1.0)) === 1.0
        @test entropy(NormalMeanVariance(1.0, 1.0))   ≈ 1.41893853320467
        @test pdf(NormalMeanVariance(1.0, 1.0), 1.0)  ≈ 0.3989422804014327
        @test pdf(NormalMeanVariance(1.0, 1.0), -1.0) ≈ 0.05399096651318806
        @test pdf(NormalMeanVariance(1.0, 1.0), 0.0)  ≈ 0.24197072451914337
        @test logpdf(NormalMeanVariance(1.0, 1.0), 1.0)  ≈ -0.9189385332046728
        @test logpdf(NormalMeanVariance(1.0, 1.0), -1.0) ≈ -2.9189385332046727
        @test logpdf(NormalMeanVariance(1.0, 1.0), 0.0)  ≈ -1.4189385332046727

        @test mean(NormalMeanVariance(1.0, 2.0))      === 1.0
        @test median(NormalMeanVariance(1.0, 2.0))    === 1.0
        @test mode(NormalMeanVariance(1.0, 2.0))      === 1.0
        @test var(NormalMeanVariance(1.0, 2.0))       === 2.0
        @test std(NormalMeanVariance(1.0, 2.0))       === sqrt(2.0)
        @test cov(NormalMeanVariance(1.0, 2.0))       === 2.0
        @test invcov(NormalMeanVariance(1.0, 2.0))    === inv(2.0)
        @test precision(NormalMeanVariance(1.0, 2.0)) === inv(2.0)
        @test entropy(NormalMeanVariance(1.0, 2.0))   ≈ 1.7655121234846454
        @test pdf(NormalMeanVariance(1.0, 2.0), 1.0)  ≈ 0.28209479177387814
        @test pdf(NormalMeanVariance(1.0, 2.0), -1.0) ≈ 0.1037768743551487
        @test pdf(NormalMeanVariance(1.0, 2.0), 0.0)  ≈ 0.21969564473386122
        @test logpdf(NormalMeanVariance(1.0, 2.0), 1.0)  ≈ -1.2655121234846454
        @test logpdf(NormalMeanVariance(1.0, 2.0), -1.0) ≈ -2.2655121234846454
        @test logpdf(NormalMeanVariance(1.0, 2.0), 0.0)  ≈ -1.5155121234846454
        
    end

    @testset "Base methods" begin
        @test convert(NormalMeanVariance{Float32}, NormalMeanVariance()) === NormalMeanVariance{Float32}(0f0, 1f0)
    end

    @testset "prod" begin
        
        @test prod(ProdPreserveParametrisation(), NormalMeanVariance(-1, 1), NormalMeanVariance(1, 1)) ≈ NormalMeanVariance(0.0, 0.5)
        @test prod(ProdPreserveParametrisation(), NormalMeanVariance(-1, 2), NormalMeanVariance(1, 4)) ≈ NormalMeanVariance(-1/3, 4/3)
        @test prod(ProdPreserveParametrisation(), NormalMeanVariance(2, 2), NormalMeanVariance(0, 10)) ≈ NormalMeanVariance(5/3, 5/3)

    end

end

end