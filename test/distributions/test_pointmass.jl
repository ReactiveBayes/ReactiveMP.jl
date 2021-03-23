module PointMassTest

using Test
using ReactiveMP
using Distributions
using Random
using SpecialFunctions

import ReactiveMP: ∞

@testset "PointMass" begin

    @testset "Real-based PointMass" begin

        scalar = rand()
        dist   = PointMass(scalar)

        @test variate_form(dist) === Univariate
        @test_throws BoundsError dist[2]
        @test_throws BoundsError dist[2, 2]

        @test insupport(dist, scalar)
        @test !insupport(dist, scalar + 1e-6)
        @test !insupport(dist, scalar - 1e-6)

        @test pdf(dist, scalar)        === 1.0
        @test pdf(dist, scalar + 1e-6) === 0.0
        @test pdf(dist, scalar - 1e-6) === 0.0

        @test logpdf(dist, scalar)        === 0.0
        @test logpdf(dist, scalar + 1e-6) === -Inf
        @test logpdf(dist, scalar - 1e-6) === -Inf

        @test_throws MethodError insupport(dist, [ 1.0, 1.0 ])
        @test_throws MethodError insupport(dist, [ 1.0 1.0; 1.0 1.0 ])
        @test_throws MethodError pdf(dist, [ 1.0, 1.0 ])
        @test_throws MethodError pdf(dist, [ 1.0 1.0; 1.0 1.0 ])
        @test_throws MethodError logpdf(dist, [ 1.0, 1.0 ])
        @test_throws MethodError logpdf(dist, [ 1.0 1.0; 1.0 1.0 ])

        @test entropy(dist) === -∞

        @test mean(dist)       === scalar
        @test var(dist)        === 0.0
        @test std(dist)        === 0.0
        @test cov(dist)        === 0.0
        @test precision(dist)  === Inf
        @test ndims(dist)      === 1

        @test_throws ErrorException probvec(dist) 
        @test logmean(dist)         === log(scalar)
        @test inversemean(dist)     === inv(scalar)
        @test mirroredlogmean(dist) === log(1.0 - scalar)
        @test loggammamean(dist)    === loggamma(scalar)
    end

    @testset "Vector-based PointMass" begin

        vector = rand(5)
        dist   = PointMass(vector)

        @test variate_form(dist) === Multivariate
        @test dist[2] === vector[2]
        @test dist[3] === vector[3]
        @test_throws BoundsError dist[6]
        @test_throws BoundsError dist[3, 5]

        @test insupport(dist, vector)
        @test !insupport(dist, vector .+ 1e-6)
        @test !insupport(dist, vector .- 1e-6)

        @test pdf(dist, vector)         === 1.0
        @test pdf(dist, vector .+ 1e-6) === 0.0
        @test pdf(dist, vector .- 1e-6) === 0.0

        @test logpdf(dist, vector)         === 0.0
        @test logpdf(dist, vector .+ 1e-6) === -Inf
        @test logpdf(dist, vector .- 1e-6) === -Inf

        @test_throws MethodError insupport(dist, 1.0)
        @test_throws MethodError insupport(dist, [ 1.0 1.0; 1.0 1.0 ])
        @test_throws MethodError pdf(dist, 1.0)
        @test_throws MethodError pdf(dist, [ 1.0 1.0; 1.0 1.0 ])
        @test_throws MethodError logpdf(dist, 1.0)
        @test_throws MethodError logpdf(dist, [ 1.0 1.0; 1.0 1.0 ])

        @test entropy(dist) === -∞

        @test mean(dist)       == vector
        @test var(dist)        == zeros(5)
        @test std(dist)        == zeros(5)
        @test cov(dist)        == zeros(5, 5)
        @test precision(dist)  == fill(Inf, (5, 5))
        @test ndims(dist)      == 5

        @test probvec(dist)      == vector
        @test logmean(dist)      == log.(vector)
        @test_throws ErrorException inversemean(dist)
        @test_throws ErrorException mirroredlogmean(dist)
        @test loggammamean(dist) == loggamma.(vector)
    end

    @testset "Matrix-based PointMass" begin

        matrix = rand(5, 5)
        dist   = PointMass(matrix)

        @test variate_form(dist) === Matrixvariate
        @test dist[2]    === matrix[2]
        @test dist[3]    === matrix[3]
        @test dist[3, 3] === matrix[3, 3]
        @test_throws BoundsError dist[100]
        @test_throws BoundsError dist[100, 100]

        @test insupport(dist, matrix)
        @test !insupport(dist, matrix .+ 1e-6)
        @test !insupport(dist, matrix .- 1e-6)

        @test pdf(dist, matrix)         === 1.0
        @test pdf(dist, matrix .+ 1e-6) === 0.0
        @test pdf(dist, matrix .- 1e-6) === 0.0

        @test logpdf(dist, matrix)         === 0.0
        @test logpdf(dist, matrix .+ 1e-6) === -Inf
        @test logpdf(dist, matrix .- 1e-6) === -Inf

        @test_throws MethodError insupport(dist, 1.0)
        @test_throws MethodError insupport(dist, [ 1.0, 1.0 ])
        @test_throws MethodError pdf(dist, 1.0)
        @test_throws MethodError pdf(dist, [ 1.0, 1.0 ])
        @test_throws MethodError logpdf(dist, 1.0)
        @test_throws MethodError logpdf(dist, [ 1.0, 1.0 ])

        @test entropy(dist) === -∞

        @test mean(dist)       == matrix
        @test var(dist)        == zeros(5, 5)
        @test std(dist)        == zeros(5, 5)
        @test ndims(dist)      == (5, 5)

        @test_throws ErrorException cov(dist)
        @test_throws ErrorException precision(dist)

        @test_throws ErrorException probvec(dist)
        @test logmean(dist)      == log.(matrix)
        @test inversemean(dist)  ≈ cholinv(matrix)
        @test_throws ErrorException mirroredlogmean(dist)
        @test loggammamean(dist) == loggamma.(matrix)
    end

end

end
