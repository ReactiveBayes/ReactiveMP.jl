module MvNormalMeanPrecisionTest

using Test
using ReactiveMP

@testset "MvNormalMeanPrecision" begin

    @testset "Constructor" begin
        @test MvNormalMeanPrecision([ 1.0, 1.0 ]) == MvNormalMeanPrecision([ 1.0, 1.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanPrecision([ 1.0, 2.0 ]) == MvNormalMeanPrecision([ 1.0, 2.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanPrecision([ 1, 2 ])     == MvNormalMeanPrecision([ 1.0, 2.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanPrecision([ 1f0, 2f0 ]) == MvNormalMeanPrecision([ 1f0, 2f0 ], [ 1f0, 1f0 ])

        @test eltype(MvNormalMeanPrecision([ 1.0, 1.0 ]))               === Float64
        @test eltype(MvNormalMeanPrecision([ 1.0, 1.0 ], [ 1.0, 1.0 ])) === Float64
        @test eltype(MvNormalMeanPrecision([ 1, 1 ]))                   === Float64
        @test eltype(MvNormalMeanPrecision([ 1, 1 ], [ 1, 1 ]))         === Float64
        @test eltype(MvNormalMeanPrecision([ 1f0, 1f0 ]))               === Float32
        @test eltype(MvNormalMeanPrecision([ 1f0, 1f0 ], [ 1f0, 1f0 ])) === Float32
    end

    @testset "Stats methods" begin
        
        μ    = [ 0.2, 3.0, 4.0 ]
        Λ    = [ 1.5 -0.3 0.1; -0.3 1.8 0.0; 0.1 0.0 3.5 ]
        dist = MvNormalMeanPrecision(μ, Λ)

        @test mean(dist)         == μ
        @test mode(dist)         == μ
        @test weightedmean(dist) == Λ * μ
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
        @test convert(MvNormalMeanPrecision{Float32}, MvNormalMeanPrecision([ 0.0, 0.0 ])) == MvNormalMeanPrecision([ 0f0, 0f0 ], [ 1f0, 1f0 ])
        @test convert(MvNormalMeanPrecision{Float64}, [ 0.0, 0.0 ], [ 2 0; 0 3 ]) == MvNormalMeanPrecision([ 0.0, 0.0 ], [ 2.0 0.0; 0.0 3.0 ])

        @test length(MvNormalMeanPrecision([ 0.0, 0.0 ]))      === 2
        @test length(MvNormalMeanPrecision([ 0.0, 0.0, 0.0 ])) === 3
        @test ndims(MvNormalMeanPrecision([ 0.0, 0.0 ]))       === 2
        @test ndims(MvNormalMeanPrecision([ 0.0, 0.0, 0.0 ]))  === 3
        @test size(MvNormalMeanPrecision([ 0.0, 0.0 ]))        === (2, )
        @test size(MvNormalMeanPrecision([ 0.0, 0.0, 0.0 ]))   === (3, )
    end

    @testset "prod" begin
        
        @test prod(ProdPreserveParametrisation(), MvNormalMeanPrecision([ -1, -1 ], [ 2, 2 ]), MvNormalMeanPrecision([ 1, 1 ], [ 2, 4 ])) ≈ MvNormalMeanPrecision([ 0, 1/3 ], [ 4, 6 ])

        μ    = [ 0.2, 3.0, 4.0 ]
        Λ    = [ 1.5 -0.1 0.1; -0.1 1.8 0.0; 0.1 0.0 3.5 ]
        dist = MvNormalMeanPrecision(μ, Λ)

        @test prod(ProdPreserveParametrisation(), dist, dist) ≈ MvNormalMeanPrecision([0.20, 3.00, 4.00], [ 3.00 -0.20 0.20; -0.20 3.60 0.00; 0.20 0.00 7.00])

    end

end

end