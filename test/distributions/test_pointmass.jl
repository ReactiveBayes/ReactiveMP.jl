module PointMassTest

using Test
using ReactiveMP
using Distributions
using Random
using SpecialFunctions

import ReactiveMP: ∞

@testset "PointMass" begin

    @testset "Scalar-based PointMass" begin

        scalar = rand()
        dist   = PointMass(scalar)

        @test variate_form(dist) === Univariate
        @test_throws BoundsError dist[2]
        @test_throws MethodError dist[2, 2]

        @test insupport(dist, scalar)
        @test !insupport(dist, scalar + 1e-6)
        @test !insupport(dist, scalar - 1e-6)

        @test pdf(dist, scalar)        === 1.0
        @test pdf(dist, scalar + 1e-6) === 0.0
        @test pdf(dist, scalar - 1e-6) === 0.0

        @test logpdf(dist, scalar)        === 0.0
        @test logpdf(dist, scalar + 1e-6) === -Inf
        @test logpdf(dist, scalar - 1e-6) === -Inf

        @test entropy(dist) === -∞

        @test mean(dist)       === scalar
        @test var(dist)        === 0.0
        @test std(dist)        === 0.0
        @test cov(dist)        === 0.0
        @test precision(dist)  === Inf
        @test ndims(dist)      === 1

        @test_throws ErrorException probvec(dist) 
        @test logmean(dist)      === log(scalar)
        @test inversemean(dist)  === inv(scalar)
        @test loggammamean(dist) === loggamma(scalar)
    end

end

end
