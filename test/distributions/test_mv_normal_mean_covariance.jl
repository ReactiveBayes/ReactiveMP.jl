module MvNormalMeanCovarianceTest

using Test
using ReactiveMP

@testset "MvNormalMeanCovariance" begin

    @testset "Constructor" begin
        @test MvNormalMeanCovariance([ 1.0, 1.0 ]) == MvNormalMeanCovariance([ 1.0, 1.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanCovariance([ 1.0, 2.0 ]) == MvNormalMeanCovariance([ 1.0, 2.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanCovariance([ 1, 2 ])     == MvNormalMeanCovariance([ 1.0, 2.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanCovariance([ 1f0, 2f0 ]) == MvNormalMeanCovariance([ 1f0, 2f0 ], [ 1f0, 1f0 ])

        @test eltype(MvNormalMeanCovariance([ 1.0, 1.0 ]))               === Float64
        @test eltype(MvNormalMeanCovariance([ 1.0, 1.0 ], [ 1.0, 1.0 ])) === Float64
        @test eltype(MvNormalMeanCovariance([ 1, 1 ]))                   === Float64
        @test eltype(MvNormalMeanCovariance([ 1, 1 ], [ 1, 1 ]))         === Float64
        @test eltype(MvNormalMeanCovariance([ 1f0, 1f0 ]))               === Float32
        @test eltype(MvNormalMeanCovariance([ 1f0, 1f0 ], [ 1f0, 1f0 ])) === Float32
    end

    @testset "Stats methods" begin
        
        μ    = [ 0.2, 3.0, 4.0 ]
        Σ    = [ 1.5 -0.3 0.1; -0.3 1.8 0.0; 0.1 0.0 3.5 ]
        dist = MvNormalMeanCovariance(μ, Σ)

        @test mean(dist)      == μ
        @test mode(dist)      == μ
        @test invcov(dist)    ≈  inv(Σ)
        @test precision(dist) ≈  inv(Σ)
        @test cov(dist)       == Σ
        @test std(dist)       ≈ sqrt(Σ)
        
        @test length(dist) == 3
        @test entropy(dist) ≈ 5.361886000915401
        @test pdf(dist, [ 0.2, 3.0, 4.0 ])         ≈ 0.021028302702542
        @test pdf(dist,  [0.202, 3.002, 4.002])    ≈ 0.021028229679079503
        @test logpdf(dist, [ 0.2, 3.0, 4.0 ])      ≈ -3.8618860009154012
        @test logpdf(dist,  [0.202, 3.002, 4.002]) ≈ -3.861889473548943
        
    end

    @testset "Base methods" begin
        @test convert(MvNormalMeanCovariance{Float32}, MvNormalMeanCovariance([ 0.0, 0.0 ])) == MvNormalMeanCovariance([ 0f0, 0f0 ], [ 1f0, 1f0 ])
    end

    @testset "prod" begin
        
        @test prod(ProdPreserveParametrisation(), MvNormalMeanCovariance([ -1, -1 ], [ 2, 2 ]), MvNormalMeanCovariance([ 1, 1 ], [ 2, 4 ])) ≈ MvNormalMeanCovariance([ 0, -1/3 ], [ 1, 4/3 ])

        μ    = [ 0.2, 3.0, 4.0 ]
        Σ    = [ 1.5 -0.1 0.1; -0.1 1.8 0.0; 0.1 0.0 3.5 ]
        dist = MvNormalMeanCovariance(μ, Σ)

        @test prod(ProdPreserveParametrisation(), dist, dist) ≈ MvNormalMeanCovariance([0.20, 3.00, 4.00], [0.75 -0.05 0.05; -0.05 0.90 6.83e-19; 0.05 6.83e-19 1.75])

    end

end

end