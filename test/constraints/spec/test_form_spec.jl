module ReactiveMPFormConstraintsSpecTest 

using Test
using ReactiveMP 
using GraphPPL

import ReactiveMP: CompositeFormConstraint
import ReactiveMP: resolve_marginal_form_prod, resolve_messages_form_prod


@testset "Form constraints specification" begin 

    # we dont need real model for form constraints resolution
    model = nothing

    @testset "Use case #1" begin
        cs = @constraints begin 
            q(x) :: PointMass
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :x) 
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #2" begin
        cs = @constraints begin 
            q(x) :: Nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :x) 
            form === UnspecifiedFormConstraint() && prod === ProdAnalytical()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #2" begin
        cs = @constraints begin 
            q(x) :: SampleList(5000, LeftProposal())
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :x) 
            typeof(form) <: SampleListFormConstraint && prod === ProdGeneric() 
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #3" begin
        cs = @constraints begin 
            q(x) :: PointMass(optimizer = "optimizer")
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :x) 
            typeof(form) <: PointMassFormConstraint && prod === ProdGeneric() 
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #4" begin
        cs = @constraints begin 
            q(x) :: SampleList(5000, LeftProposal()) :: PointMass(optimizer = "optimizer")
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :x) 
            typeof(form) <: CompositeFormConstraint && prod === ProdGeneric() 
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #5" begin
        @constraints function cs5(flag)
            if flag 
                q(x) :: PointMass
            end
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(true), model, :x) 
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(true), model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(false), model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(false), model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(true), model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(false), model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(true), model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(false), model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #6" begin
        cs = @constraints begin 
            μ(x) :: PointMass
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :x) 
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #7" begin
        cs = @constraints begin 
            μ(x) :: Nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :x) 
            form === UnspecifiedFormConstraint() && prod === ProdAnalytical()
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #8" begin
        cs = @constraints begin 
            μ(x) :: SampleList(5000, LeftProposal())
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :x) 
            typeof(form) <: SampleListFormConstraint && prod === ProdGeneric() 
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #9" begin
        cs = @constraints begin 
            μ(x) :: PointMass(optimizer = "optimizer")
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :x) 
            typeof(form) <: PointMassFormConstraint && prod === ProdGeneric() 
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #10" begin
        cs = @constraints begin 
            μ(x) :: SampleList(5000, LeftProposal()) :: PointMass(optimizer = "optimizer")
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :x) 
            typeof(form) <: CompositeFormConstraint && prod === ProdGeneric() 
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #11" begin
        @constraints function cs5(flag)
            if flag 
                μ(x) :: PointMass
            end
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(true), model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(true), model, :x) 
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(false), model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(false), model, :x) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(true), model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(true), model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs5(false), model, :y) 
            form === nothing && prod === nothing
        end

        @test let (form, prod) = resolve_messages_form_prod(cs5(false), model, :y) 
            form === nothing && prod === nothing
        end
    end

    @testset "Use case #12" begin 
        cs = @constraints begin 
            q(x) :: PointMass
            μ(x) :: SampleList(5000, LeftProposal())
            q(y) :: Nothing 
            μ(y) :: PointMass
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :x) 
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :x) 
            typeof(form) <: SampleListFormConstraint && prod === ProdGeneric()
        end

        @test let (form, prod) = resolve_marginal_form_prod(cs, model, :y) 
            form === UnspecifiedFormConstraint() && prod === ProdAnalytical()
        end

        @test let (form, prod) = resolve_messages_form_prod(cs, model, :y) 
            form === PointMassFormConstraint() && prod === ProdGeneric()
        end
    end

    @testset "Error case #1" begin
        @test_throws ErrorException @constraints begin 
            q(x) :: Nothing 
            q(x) :: PointMass
        end

        @test_throws ErrorException @constraints begin 
            μ(x) :: Nothing 
            μ(x) :: PointMass
        end
    end
    
end

end