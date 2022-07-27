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
        @test ReactiveMP.contingency_matrix(Contingency(ones(3, 3))) == ones(3, 3) ./ 9
        @test ReactiveMP.contingency_matrix(Contingency(ones(3, 3), Val(true))) == ones(3, 3) ./ 9
        @test ReactiveMP.contingency_matrix(Contingency(ones(3, 3), Val(false))) == ones(3, 3) # Matrix is wrong, but just to test that `false` is working
        @test ReactiveMP.contingency_matrix(Contingency(ones(4, 4))) == ones(4, 4) ./ 16
        @test ReactiveMP.contingency_matrix(Contingency(ones(4, 4), Val(true))) == ones(4, 4) ./ 16
        @test ReactiveMP.contingency_matrix(Contingency(ones(4, 4), Val(false))) == ones(4, 4)
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
        @test entropy(Contingency([0.7 0.1; 0.1 0.1])) ≈ 0.9404479886553263
        @test entropy(Contingency(10.0 * [0.7 0.1; 0.1 0.1])) ≈ 0.9404479886553263
        @test entropy(Contingency([0.07 0.41; 0.31 0.21])) ≈ 1.242506182893139
        @test entropy(Contingency(10.0 * [0.07 0.41; 0.31 0.21])) ≈ 1.242506182893139
        @test entropy(Contingency([0.09 0.00; 0.00 0.91])) ≈ 0.30253782309749805
        @test entropy(Contingency(10.0 * [0.09 0.00; 0.00 0.91])) ≈ 0.30253782309749805
    end
end

end
