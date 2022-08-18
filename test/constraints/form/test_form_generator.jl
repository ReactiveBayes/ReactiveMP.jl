module ReactiveMPGeneratorFormConstraintTest

using Test
using ReactiveMP
using Random
using LinearAlgebra
using StableRNGs

import ReactiveMP: FormConstraintsGenerator, PointMassFormConstraint, AbstractFormConstraint

@testset "GeneratorFormConstraint" begin

    @testset "marginal form constraints resolution" begin 
        mutable struct MyCustomMarginalFormConstraint <: AbstractFormConstraint
            some_field :: Int
        end

        constraint           = MyCustomMarginalFormConstraint(0)
        generator_constraint = FormConstraintsGenerator(() -> MyCustomMarginalFormConstraint(0))

        model = FactorGraphModel()

        xoptions = ReactiveMP.RandomVariableCreationOptions()
        xoptions = ReactiveMP.randomvar_options_set_marginal_form_constraint(constraint)

        x = randomvar(model, xoptions, :x, 2)

        x1_q_constraint = ReactiveMP.marginal_form_constraint(x[1])
        x2_q_constraint = ReactiveMP.marginal_form_constraint(x[2])

        @test x1_q_constraint isa MyCustomMarginalFormConstraint
        @test x2_q_constraint isa MyCustomMarginalFormConstraint
        @test x1_q_constraint === x2_q_constraint

        yoptions = ReactiveMP.RandomVariableCreationOptions()
        yoptions = ReactiveMP.randomvar_options_set_marginal_form_constraint(generator_constraint)

        y = randomvar(model, yoptions, :y, 2)

        y1_q_constraint = ReactiveMP.marginal_form_constraint(y[1])
        y2_q_constraint = ReactiveMP.marginal_form_constraint(y[2])

        @test y1_q_constraint isa MyCustomMarginalFormConstraint
        @test y2_q_constraint isa MyCustomMarginalFormConstraint
        @test y1_q_constraint !== y2_q_constraint
    end

    @testset "messages form constraints resolution" begin 
        mutable struct MyCustomMessagesFormConstraint <: AbstractFormConstraint
            some_field :: Int
        end

        constraint           = MyCustomMessagesFormConstraint(0)
        generator_constraint = FormConstraintsGenerator(() -> MyCustomMessagesFormConstraint(0))

        model = FactorGraphModel()

        xoptions = ReactiveMP.RandomVariableCreationOptions()
        xoptions = ReactiveMP.randomvar_options_set_messages_form_constraint(constraint)

        x = randomvar(model, xoptions, :x, 2)

        x1_q_constraint = ReactiveMP.messages_form_constraint(x[1])
        x2_q_constraint = ReactiveMP.messages_form_constraint(x[2])

        @test x1_q_constraint isa MyCustomMessagesFormConstraint
        @test x2_q_constraint isa MyCustomMessagesFormConstraint
        @test x1_q_constraint === x2_q_constraint

        yoptions = ReactiveMP.RandomVariableCreationOptions()
        yoptions = ReactiveMP.randomvar_options_set_messages_form_constraint(generator_constraint)

        y = randomvar(model, yoptions, :y, 2)

        y1_q_constraint = ReactiveMP.messages_form_constraint(y[1])
        y2_q_constraint = ReactiveMP.messages_form_constraint(y[2])

        @test y1_q_constraint isa MyCustomMessagesFormConstraint
        @test y2_q_constraint isa MyCustomMessagesFormConstraint
        @test y1_q_constraint !== y2_q_constraint
    end

end

end