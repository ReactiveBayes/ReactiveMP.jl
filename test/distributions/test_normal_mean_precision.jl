module NormalMeanPrecisionTest

using Test
using ReactiveMP

@testset "NormalMeanPrecision" begin

    @testset "Constructor" begin
        @test NormalMeanPrecision()         == NormalMeanPrecision{Float64}(0.0, 1.0)
        @test NormalMeanPrecision(1.0)      == NormalMeanPrecision{Float64}(1.0, 1.0)
        @test NormalMeanPrecision(1.0, 2.0) == NormalMeanPrecision{Float64}(1.0, 2.0)
        @test NormalMeanPrecision(1)        == NormalMeanPrecision{Float64}(1.0, 1.0)
        @test NormalMeanPrecision(1, 2)     == NormalMeanPrecision{Float64}(1.0, 2.0)
        @test NormalMeanPrecision(1f0)      == NormalMeanPrecision{Float32}(1f0, 1f0)
        @test NormalMeanPrecision(1f0, 2f0) == NormalMeanPrecision{Float32}(1f0, 2f0)

        @test eltype(NormalMeanPrecision())         === Float64
        @test eltype(NormalMeanPrecision(0.0))      === Float64
        @test eltype(NormalMeanPrecision(0.0, 1.0)) === Float64
        @test eltype(NormalMeanPrecision(0))        === Float64
        @test eltype(NormalMeanPrecision(0, 1))     === Float64
        @test eltype(NormalMeanPrecision(0f0))      === Float32
        @test eltype(NormalMeanPrecision(0f0, 1f0)) === Float32
    end

    @testset "Stats methods" begin
        
        @test mean(NormalMeanPrecision(0.0, 1.0))      === 0.0
        @test median(NormalMeanPrecision(0.0, 1.0))    === 0.0
        @test mode(NormalMeanPrecision(0.0, 1.0))      === 0.0
        @test var(NormalMeanPrecision(0.0, 1.0))       === 1.0
        @test std(NormalMeanPrecision(0.0, 1.0))       === 1.0
        @test cov(NormalMeanPrecision(0.0, 1.0))       === 1.0
        @test invcov(NormalMeanPrecision(0.0, 1.0))    === 1.0
        @test precision(NormalMeanPrecision(0.0, 1.0)) === 1.0
        @test entropy(NormalMeanPrecision(0.0, 1.0))   ≈ 1.41893853320467
        @test pdf(NormalMeanPrecision(0.0, 1.0), 1.0)  ≈ 0.24197072451914337
        @test pdf(NormalMeanPrecision(0.0, 1.0), -1.0) ≈ 0.24197072451914337
        @test pdf(NormalMeanPrecision(0.0, 1.0), 0.0)  ≈ 0.3989422804014327
        @test logpdf(NormalMeanPrecision(0.0, 1.0), 1.0)  ≈ -1.4189385332046727
        @test logpdf(NormalMeanPrecision(0.0, 1.0), -1.0) ≈ -1.4189385332046727
        @test logpdf(NormalMeanPrecision(0.0, 1.0), 0.0)  ≈ -0.9189385332046728

        @test mean(NormalMeanPrecision(1.0, 1.0))      === 1.0
        @test median(NormalMeanPrecision(1.0, 1.0))    === 1.0
        @test mode(NormalMeanPrecision(1.0, 1.0))      === 1.0
        @test var(NormalMeanPrecision(1.0, 1.0))       === 1.0
        @test std(NormalMeanPrecision(1.0, 1.0))       === 1.0
        @test cov(NormalMeanPrecision(1.0, 1.0))       === 1.0
        @test invcov(NormalMeanPrecision(1.0, 1.0))    === 1.0
        @test precision(NormalMeanPrecision(1.0, 1.0)) === 1.0
        @test entropy(NormalMeanPrecision(1.0, 1.0))   ≈ 1.41893853320467
        @test pdf(NormalMeanPrecision(1.0, 1.0), 1.0)  ≈ 0.3989422804014327
        @test pdf(NormalMeanPrecision(1.0, 1.0), -1.0) ≈ 0.05399096651318806
        @test pdf(NormalMeanPrecision(1.0, 1.0), 0.0)  ≈ 0.24197072451914337
        @test logpdf(NormalMeanPrecision(1.0, 1.0), 1.0)  ≈ -0.9189385332046728
        @test logpdf(NormalMeanPrecision(1.0, 1.0), -1.0) ≈ -2.9189385332046727
        @test logpdf(NormalMeanPrecision(1.0, 1.0), 0.0)  ≈ -1.4189385332046727

        @test mean(NormalMeanPrecision(1.0, 0.5))      === 1.0
        @test median(NormalMeanPrecision(1.0, 0.5))    === 1.0
        @test mode(NormalMeanPrecision(1.0, 0.5))      === 1.0
        @test var(NormalMeanPrecision(1.0, 0.5))       === 2.0
        @test std(NormalMeanPrecision(1.0, 0.5))       === sqrt(2.0)
        @test cov(NormalMeanPrecision(1.0, 0.5))       === 2.0
        @test invcov(NormalMeanPrecision(1.0, 0.5))    === inv(2.0)
        @test precision(NormalMeanPrecision(1.0, 0.5)) === inv(2.0)
        @test entropy(NormalMeanPrecision(1.0, 0.5))   ≈ 1.7655121234846454
        @test pdf(NormalMeanPrecision(1.0, 0.5), 1.0)  ≈ 0.28209479177387814
        @test pdf(NormalMeanPrecision(1.0, 0.5), -1.0) ≈ 0.1037768743551487
        @test pdf(NormalMeanPrecision(1.0, 0.5), 0.0)  ≈ 0.21969564473386122
        @test logpdf(NormalMeanPrecision(1.0, 0.5), 1.0)  ≈ -1.2655121234846454
        @test logpdf(NormalMeanPrecision(1.0, 0.5), -1.0) ≈ -2.2655121234846454
        @test logpdf(NormalMeanPrecision(1.0, 0.5), 0.0)  ≈ -1.5155121234846454
        
    end

    @testset "Base methods" begin
        @test convert(NormalMeanPrecision{Float32}, NormalMeanPrecision()) === NormalMeanPrecision{Float32}(0f0, 1f0)
    end

    @testset "prod" begin
        
        @test prod(ProdPreserveParametrisation(), NormalMeanPrecision(-1, 1/1), NormalMeanPrecision(1, 1/1)) ≈ NormalMeanPrecision(0.0, 2.0)
        @test prod(ProdPreserveParametrisation(), NormalMeanPrecision(-1, 1/2), NormalMeanPrecision(1, 1/4)) ≈ NormalMeanPrecision(-1/3, 3/4)
        @test prod(ProdPreserveParametrisation(), NormalMeanPrecision(2, 1/2), NormalMeanPrecision(0, 1/10)) ≈ NormalMeanPrecision(5/3, 3/5)

    end

end

end