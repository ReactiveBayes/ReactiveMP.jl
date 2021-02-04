module MvNormalWeightedMeanPrecisionTest

using Test
using ReactiveMP

@testset "MvNormalWeightedMeanPrecision" begin

    @testset "Constructor" begin
        @test MvNormalWeightedMeanPrecision([ 1.0, 1.0 ]) == MvNormalWeightedMeanPrecision([ 1.0, 1.0 ], [ 1.0, 1.0 ])
        @test MvNormalWeightedMeanPrecision([ 1.0, 2.0 ]) == MvNormalWeightedMeanPrecision([ 1.0, 2.0 ], [ 1.0, 1.0 ])
        @test MvNormalWeightedMeanPrecision([ 1, 2 ])     == MvNormalWeightedMeanPrecision([ 1.0, 2.0 ], [ 1.0, 1.0 ])
        @test MvNormalWeightedMeanPrecision([ 1f0, 2f0 ]) == MvNormalWeightedMeanPrecision([ 1f0, 2f0 ], [ 1f0, 1f0 ])

        @test eltype(MvNormalWeightedMeanPrecision([ 1.0, 1.0 ]))               === Float64
        @test eltype(MvNormalWeightedMeanPrecision([ 1.0, 1.0 ], [ 1.0, 1.0 ])) === Float64
        @test eltype(MvNormalWeightedMeanPrecision([ 1, 1 ]))                   === Float64
        @test eltype(MvNormalWeightedMeanPrecision([ 1, 1 ], [ 1, 1 ]))         === Float64
        @test eltype(MvNormalWeightedMeanPrecision([ 1f0, 1f0 ]))               === Float32
        @test eltype(MvNormalWeightedMeanPrecision([ 1f0, 1f0 ], [ 1f0, 1f0 ])) === Float32
    end

    @testset "Stats methods" begin
        
        xi   = [ -0.2, 5.34, 14.02 ]
        Λ    = [ 1.5 -0.3 0.1; -0.3 1.8 0.0; 0.1 0.0 3.5 ]
        dist = MvNormalWeightedMeanPrecision(xi, Λ)

        @test mean(dist)         == cholinv(Λ) * xi
        @test mode(dist)         == cholinv(Λ) * xi
        @test weightedmean(dist) == xi
        @test invcov(dist)       == Λ
        @test precision(dist)    == Λ
        @test cov(dist)          ≈ cholinv(Λ)
        @test std(dist)          ≈ cholsqrt(cholinv(Λ))
        
        @test length(dist) == 3
        @test entropy(dist) ≈ 3.1517451983126357
        @test pdf(dist, [ 0.2, 3.0, 4.0 ])         ≈ 0.19171503573907536
        @test pdf(dist,  [0.202, 3.002, 4.002])    ≈ 0.19171258180232315
        @test logpdf(dist, [ 0.2, 3.0, 4.0 ])      ≈ -1.6517451983126357
        @test logpdf(dist,  [0.202, 3.002, 4.002]) ≈ -1.6517579983126356
        
    end

    @testset "Base methods" begin
        @test convert(MvNormalWeightedMeanPrecision{Float32}, MvNormalWeightedMeanPrecision([ 0.0, 0.0 ])) == MvNormalWeightedMeanPrecision([ 0f0, 0f0 ], [ 1f0, 1f0 ])
        @test convert(MvNormalWeightedMeanPrecision{Float64}, [ 0.0, 0.0 ], [ 2 0; 0 3 ]) == MvNormalWeightedMeanPrecision([ 0.0, 0.0 ], [ 2.0 0.0; 0.0 3.0 ])

        @test length(MvNormalWeightedMeanPrecision([ 0.0, 0.0 ]))      === 2
        @test length(MvNormalWeightedMeanPrecision([ 0.0, 0.0, 0.0 ])) === 3
        @test ndims(MvNormalWeightedMeanPrecision([ 0.0, 0.0 ]))       === 2
        @test ndims(MvNormalWeightedMeanPrecision([ 0.0, 0.0, 0.0 ]))  === 3
        @test size(MvNormalWeightedMeanPrecision([ 0.0, 0.0 ]))        === (2, )
        @test size(MvNormalWeightedMeanPrecision([ 0.0, 0.0, 0.0 ]))   === (3, )
    end

    @testset "prod" begin
        
        @test prod(ProdPreserveParametrisation(), MvNormalWeightedMeanPrecision([ -1, -1 ], [ 2, 2 ]), MvNormalWeightedMeanPrecision([ 1, 1 ], [ 2, 4 ])) ≈ MvNormalWeightedMeanPrecision([ 0, 0 ], [ 4, 6 ])

        xi   = [ 0.2, 3.0, 4.0 ]
        Λ    = [ 1.5 -0.1 0.1; -0.1 1.8 0.0; 0.1 0.0 3.5 ]
        dist = MvNormalWeightedMeanPrecision(xi, Λ)

        @test prod(ProdPreserveParametrisation(), dist, dist) ≈ MvNormalWeightedMeanPrecision([0.40, 6.00, 8.00], [ 3.00 -0.20 0.20; -0.20 3.60 0.00; 0.20 0.00 7.00])

    end

end

end