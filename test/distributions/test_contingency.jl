module ContingencyTest

using Test
using ReactiveMP
using Distributions
using Random

@testset "Contingency" begin
    @testset "common" begin
        @test Contingency <: Distribution
        @test Contingency <: ContinuousDistribution
        @test Contingency <: MatrixDistribution

        @test value_support(Contingency) === Continuous
        @test variate_form(Contingency) === Matrixvariate
    end

    @testset "contingency_matrix" begin
        @test ReactiveMP.contingency_matrix(Contingency(ones(3, 3))) == ones(3, 3)
        @test ReactiveMP.contingency_matrix(Contingency(ones(4, 4))) == ones(4, 4)
    end

    @testset "vague" begin
        @test_throws MethodError vague(Contingency)

        d1 = vague(Contingency, 3)

        @test typeof(d1) <: Contingency
        @test ReactiveMP.contingency_matrix(d1) ≈ ones(3, 3) ./ 9

        d2 = vague(Contingency, 4)

        @test typeof(d2) <: Contingency
        @test ReactiveMP.contingency_matrix(d2) ≈ ones(4, 4) ./ 16
    end

    @testset "entropy" begin
        @test entropy(Contingency([0.1 0.9; 0.9 0.1])) ≈ 0.6501659467828964
        @test entropy(Contingency([0.2 0.8; 0.8 0.2])) ≈ 1.0008048470763757
        @test entropy(Contingency([0.45 0.75; 0.55 0.25])) ≈ 1.2504739583323967
        @test !isnan(entropy(Contingency([0.0 1.0; 1.0 0.0])))
        @test !isinf(entropy(Contingency([0.0 1.0; 1.0 0.0])))
    end
end

end
