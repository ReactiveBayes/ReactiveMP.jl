module ReactiveMPPointMassFormConstraintTest

using Test
using ReactiveMP
using Random
using LinearAlgebra

import ReactiveMP:
    PointMassFormConstraint, is_point_mass_form_constraint, call_boundaries, call_starting_point, call_optimizer

@testset "PointMassFormConstraint" begin
    @testset "is_point_mass_form_constraint" begin
        @test is_point_mass_form_constraint(PointMassFormConstraint())
    end

    @testset "boundaries" begin
        constraint = PointMassFormConstraint()

        @test call_boundaries(constraint, NormalMeanVariance(0, 1)) === (-Inf, Inf)
        @test call_boundaries(constraint, Gamma(1, 1)) === (0.0, Inf)

        bm_constraint = PointMassFormConstraint(boundaries = (args...) -> (-1.0, 1.0))

        @test call_boundaries(bm_constraint, NormalMeanVariance(0, 1)) === (-1.0, 1.0)
        @test call_boundaries(bm_constraint, Gamma(1, 1)) === (-1.0, 1.0)
    end

    @testset "starting point" begin
        constraint = PointMassFormConstraint()

        @test call_starting_point(constraint, NormalMeanVariance(0, 1)) == [0.0]
        @test_throws ErrorException call_starting_point(constraint, Gamma(1, 1))

        bm_constraint = PointMassFormConstraint(starting_point = (args...) -> [1.0])

        @test call_starting_point(bm_constraint, NormalMeanVariance(0, 1)) == [1.0]
        @test call_starting_point(bm_constraint, Gamma(1, 1)) == [1.0]
    end

    @testset "optimizer" begin
        constraint = PointMassFormConstraint()

        @test isapprox(mean(call_optimizer(constraint, NormalMeanVariance(0, 1))), 0.0, atol = 0.1)
        @test isapprox(mean(call_optimizer(constraint, NormalMeanVariance(-10, 10))), -10.0, atol = 0.1)
        @test_throws ErrorException call_optimizer(constraint, GammaShapeRate(1, 1))

        gopt_constraint = PointMassFormConstraint(starting_point = (args...) -> [1.0])

        @test isapprox(mean(call_optimizer(gopt_constraint, GammaShapeRate(100, 10))), 10, atol = 0.1)
        @test isapprox(mean(call_optimizer(gopt_constraint, GammaShapeRate(100, 100))), 1, atol = 0.1)

        bm_constraint = PointMassFormConstraint(optimizer = (args...) -> PointMass(10.0))

        @test call_optimizer(bm_constraint, NormalMeanVariance(0, 1)) == PointMass(10.0)
        @test call_optimizer(bm_constraint, Gamma(1, 1)) == PointMass(10.0)
    end
end

end
