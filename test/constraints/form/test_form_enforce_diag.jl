module ReactiveMPEnforceDiagFormConstraintTest

using Test
using ReactiveMP
using Random
using LinearAlgebra
using StableRNGs
using DomainSets

import ReactiveMP:
    EnforceDiagFormConstraint, is_point_mass_form_constraint, __drop_offdiag, constrain_form,
    MvNormalMeanCovariance, WishartMessage, InverseWishartMessage

@testset "EnforceDiagFormConstraint" begin
    constraint = EnforceDiagFormConstraint()
    
    @testset "is_point_mass_form_constraint" begin
        @test is_point_mass_form_constraint(EnforceDiagFormConstraint()) == false
    end

    @testset "drop_offdiagonals" begin
        @test __drop_offdiag([1.0 1.0;1.0 1.0]) == [1.0 0.0; 0.0 1.0]
        @test typeof(__drop_offdiag([1.0 1.0;1.0 1.0])) == Matrix{Float64}
    end

    @testset "which_forms" begin

        N = MvNormalMeanCovariance(zeros(2), [2. 1.;1. 2.])
        W = WishartMessage(3, [2. 1.;1. 2.])
        i = InverseWishartMessage(3, [2. 1.;1. 2.])

        @test constrain_form(constraint, N) == MvNormalMeanCovariance(zeros(2), [2. 1.;1. 2.])
        @test constrain_form(constraint, W) == WishartMessage(3, [2. 0.;0. 2.])
        @test constrain_form(constraint, i) == InverseWishartMessage(3, [2. 0.;0. 2.])
        
    end
end

end
