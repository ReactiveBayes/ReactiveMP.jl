module PointMassTest

using Test
using ReactiveMP
using Distributions
using Random
using SpecialFunctions

import ReactiveMP: InfCountingReal, tiny, huge
import ReactiveMP.MacroHelpers: @test_inferred
import ReactiveMP: xtlog, mirrorlog

@testset "PointMass" begin

    rng = MersenneTwister(1234)

    @testset "Real-based PointMass" begin
        for T in (Float16, Float32, Float64, BigFloat)
            scalar = rand(rng, T)
            dist   = PointMass(scalar)

            @test variate_form(dist) === Univariate
            @test_throws BoundsError dist[2]
            @test_throws BoundsError dist[2, 2]

            @test insupport(dist, scalar)
            @test !insupport(dist, scalar + tiny)
            @test !insupport(dist, scalar - tiny)

            @test @test_inferred(T, pdf(dist, scalar))        == one(T)
            @test @test_inferred(T, pdf(dist, scalar + tiny)) == zero(T)
            @test @test_inferred(T, pdf(dist, scalar - tiny)) == zero(T)

            @test @test_inferred(T, logpdf(dist, scalar))        == zero(T)
            @test @test_inferred(T, logpdf(dist, scalar + tiny)) == convert(T, -Inf)
            @test @test_inferred(T, logpdf(dist, scalar - tiny)) == convert(T, -Inf)

            @test_throws MethodError insupport(dist, ones(T, 2))
            @test_throws MethodError insupport(dist, ones(T, 2, 2))
            @test_throws MethodError pdf(dist, ones(T, 2))
            @test_throws MethodError pdf(dist, ones(T, 2, 2))
            @test_throws MethodError logpdf(dist, ones(T, 2))
            @test_throws MethodError logpdf(dist, ones(T, 2, 2))

            @test (@inferred entropy(dist)) == InfCountingReal(eltype(dist), -1)

            @test @test_inferred(T, mean(dist))         == scalar
            @test @test_inferred(T, var(dist))          == zero(T)
            @test @test_inferred(T, std(dist))          == zero(T)
            @test @test_inferred(T, cov(dist))          == zero(T)
            @test @test_inferred(T, precision(dist))    == convert(T, Inf)
            @test @test_inferred(Int, ndims(dist))      == 1
            @test @test_inferred(Type{T}, eltype(dist)) == T

            @test_throws ErrorException probvec(dist) 
            @test @test_inferred(T, mean(log, dist))       == log(scalar)
            @test @test_inferred(T, mean(inv, dist))       == inv(scalar)
            @test @test_inferred(T, mean(mirrorlog, dist)) == log(one(scalar) - scalar)
            @test @test_inferred(T, mean(loggamma, dist))  == loggamma(scalar)
        end
    end

    @testset "Vector-based PointMass" begin
        for T in (Float16, Float32, Float64, BigFloat), N in (5, 10)
            vector = rand(rng, T, N)
            dist   = PointMass(vector)

            @test variate_form(dist) === Multivariate
            @test dist[2] === vector[2]
            @test dist[3] === vector[3]
            @test_throws BoundsError dist[N + 1]
            @test_throws BoundsError dist[N - 1, N - 1]

            @test insupport(dist, vector)
            @test !insupport(dist, vector .+ tiny)
            @test !insupport(dist, vector .- tiny)

            @test @test_inferred(T, pdf(dist, vector))         == one(T)
            @test @test_inferred(T, pdf(dist, vector .+ tiny)) == zero(T)
            @test @test_inferred(T, pdf(dist, vector .- tiny)) == zero(T)

            @test @test_inferred(T, logpdf(dist, vector))         == zero(T)
            @test @test_inferred(T, logpdf(dist, vector .+ tiny)) == convert(T, -Inf)
            @test @test_inferred(T, logpdf(dist, vector .- tiny)) == convert(T, -Inf)

            @test_throws MethodError insupport(dist, one(T))
            @test_throws MethodError insupport(dist, ones(T, 2, 2))
            @test_throws MethodError pdf(dist, one(T))
            @test_throws MethodError pdf(dist, ones(T, 2, 2))
            @test_throws MethodError logpdf(dist, one(T))
            @test_throws MethodError logpdf(dist, ones(T, 2, 2))

            @test (@inferred entropy(dist)) == InfCountingReal(eltype(dist), -1)

            @test @test_inferred(AbstractVector{T}, mean(dist))         == vector
            @test @test_inferred(AbstractVector{T}, var(dist))          == zeros(T, N)
            @test @test_inferred(AbstractVector{T}, std(dist))          == zeros(T, N)
            @test @test_inferred(AbstractMatrix{T}, cov(dist))          == zeros(T, N, N)
            @test @test_inferred(AbstractMatrix{T}, precision(dist))    == fill(convert(T, Inf), (N, N))
            @test @test_inferred(Int, ndims(dist))      == N
            @test @test_inferred(Type{T}, eltype(dist)) == T

            @test @test_inferred(AbstractVector{T}, probvec(dist))      == vector
            @test @test_inferred(AbstractVector{T}, mean(log, dist))    == log.(vector)
            @test_throws ErrorException mean(inv, dist)
            @test_throws ErrorException mean(mirrorlog, dist)
            @test @test_inferred(AbstractVector{T}, mean(loggamma, dist)) == loggamma.(vector)
        end
    end

    @testset "Matrix-based PointMass" begin
        for T in (Float16, Float32, Float64, BigFloat), N in (5, 10)
            matrix = rand(rng, T, N, N)
            dist   = PointMass(matrix)

            @test variate_form(dist) === Matrixvariate
            @test dist[2]    === matrix[2]
            @test dist[3]    === matrix[3]
            @test dist[3, 3] === matrix[3, 3]
            @test_throws BoundsError dist[N ^ 3]
            @test_throws BoundsError dist[N + 1, N + 1]

            @test insupport(dist, matrix)
            @test !insupport(dist, matrix .+ tiny)
            @test !insupport(dist, matrix .- tiny)

            @test @test_inferred(T, pdf(dist, matrix))         == one(T)
            @test @test_inferred(T, pdf(dist, matrix .+ tiny)) == zero(T)
            @test @test_inferred(T, pdf(dist, matrix .- tiny)) == zero(T)

            @test @test_inferred(T, logpdf(dist, matrix))         == zero(T)
            @test @test_inferred(T, logpdf(dist, matrix .+ tiny)) == convert(T, -Inf)
            @test @test_inferred(T, logpdf(dist, matrix .- tiny)) == convert(T, -Inf)

            @test_throws MethodError insupport(dist, one(T))
            @test_throws MethodError insupport(dist, ones(T, 2))
            @test_throws MethodError pdf(dist, one(T))
            @test_throws MethodError pdf(dist, ones(T, 2))
            @test_throws MethodError logpdf(dist, one(T))
            @test_throws MethodError logpdf(dist, ones(T, 2))

            @test (@inferred entropy(dist)) == InfCountingReal(eltype(dist), -1)

            @test @test_inferred(AbstractMatrix{T}, mean(dist))       == matrix
            @test @test_inferred(AbstractMatrix{T}, var(dist))        == zeros(N, N)
            @test @test_inferred(AbstractMatrix{T}, std(dist))        == zeros(N, N)
            @test @test_inferred(Tuple{Int, Int}, ndims(dist)) == (N, N)
            @test @test_inferred(Type{T}, eltype(dist))        == T

            @test_throws ErrorException cov(dist)
            @test_throws ErrorException precision(dist)

            @test_throws ErrorException probvec(dist)
            @test @test_inferred(AbstractMatrix{T}, mean(log, dist))  == log.(matrix)
            @test @test_inferred(AbstractMatrix{T}, mean(inv, dist))  ≈ cholinv(matrix)
            @test_throws ErrorException mean(mirrorlog, dist)
            @test @test_inferred(AbstractMatrix{T}, mean(loggamma, dist)) == loggamma.(matrix)
        end
    end

end

end
