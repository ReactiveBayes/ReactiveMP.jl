module ReactiveMPRandomvVariableTest

using Test
using ReactiveMP
using Rocket

import ReactiveMP: UnspecifiedFormConstraint
import ReactiveMP: collection_type, VariableIndividual, VariableVector, VariableArray, linear_index
import ReactiveMP: prod_constraint, prod_strategy
import ReactiveMP: proxy_variables, israndom, isproxy
import ReactiveMP: marginal_form_constraint, marginal_form_check_strategy
import ReactiveMP: messages_form_constraint, messages_form_check_strategy

@testset "RandomVariable" begin

    @testset "Simple creation" begin 

        for sym in (:x, :y, :z)
            v = randomvar(sym)

            @test israndom(v) 
            @test name(v) === sym
            @test collection_type(v) isa VariableIndividual
            @test marginal_form_constraint(v) isa UnspecifiedFormConstraint
            @test messages_form_constraint(v) isa UnspecifiedFormConstraint
            @test proxy_variables(v) === nothing
            @test prod_constraint(v) isa ProdAnalytical
            @test prod_strategy(v) isa FoldLeftProdStrategy
            @test !isproxy(v)
        end

        for sym in (:x, :y, :z), n in (10, 20)
            vs = randomvar(sym, n)
            
            @test israndom(vs) 
            @test length(vs) === n
            @test vs isa Vector
            @test all(v -> israndom(v), vs)
            @test all(v -> name(v) === sym, vs)
            @test all(v -> collection_type(v) isa VariableVector, vs)
            @test all(t -> linear_index(collection_type(t[2])) === t[1], enumerate(vs))
            @test all(v -> marginal_form_constraint(v) isa UnspecifiedFormConstraint, vs)
            @test all(v -> messages_form_constraint(v) isa UnspecifiedFormConstraint, vs)
            @test all(v -> proxy_variables(v) === nothing, vs)
            @test all(v -> prod_constraint(v) isa ProdAnalytical, vs)
            @test all(v -> prod_strategy(v) isa FoldLeftProdStrategy, vs)
            @test !isproxy(vs)
            @test all(v -> !isproxy(v), vs)
        end

        for sym in (:x, :y, :z), l in (10, 20), r in (10, 20)
            for vs in (randomvar(sym, l, r), randomvar(sym, (l, r)))
                @test israndom(vs) 
                @test size(vs) === (l, r)
                @test length(vs) === l * r
                @test vs isa Matrix
                @test all(v -> israndom(v), vs)
                @test all(v -> name(v) === sym, vs)
                @test all(v -> collection_type(v) isa VariableArray, vs)
                @test all(t -> linear_index(collection_type(t[2])) === t[1], enumerate(vs))
                @test all(v -> marginal_form_constraint(v) isa UnspecifiedFormConstraint, vs)
                @test all(v -> messages_form_constraint(v) isa UnspecifiedFormConstraint, vs)
                @test all(v -> proxy_variables(v) === nothing, vs)
                @test all(v -> prod_constraint(v) isa ProdAnalytical, vs)
                @test all(v -> prod_strategy(v) isa FoldLeftProdStrategy, vs)
                @test !isproxy(vs)
                @test all(v -> !isproxy(v), vs)
            end
        end


    end

    @testset "Creation via options" begin

        test_var     = randomvar(:tmp)
        test_options = RandomVariableCreationOptions(
            LoggerPipelineStage(),    
            (test_var, ),             
            ProdGeneric(),            
            FoldRightProdStrategy(),  
            PointMassFormConstraint(),
            FormConstraintCheckEach(),
            SampleListFormConstraint(5000, LeftProposal()),
            FormConstraintCheckLast() 
        )

        for sym in (:x, :y, :z)
            v = randomvar(test_options, sym)

            @test israndom(v) 
            @test name(v) === sym
            @test collection_type(v) isa VariableIndividual
            @test marginal_form_constraint(v) == test_options.marginal_form_constraint
            @test marginal_form_check_strategy(v) == test_options.marginal_form_check_strategy
            @test messages_form_constraint(v) == test_options.messages_form_constraint
            @test messages_form_check_strategy(v) == test_options.messages_form_check_strategy
            @test proxy_variables(v) == test_options.proxy_variables
            @test prod_constraint(v) == test_options.prod_constraint
            @test prod_strategy(v) == test_options.prod_strategy
        end

        for sym in (:x, :y, :z), n in (10, 20)
            vs = randomvar(test_options, sym, n)
            
            @test israndom(vs) 
            @test length(vs) === n
            @test vs isa Vector
            @test all(v -> israndom(v), vs)
            @test all(v -> name(v) === sym, vs)
            @test all(v -> collection_type(v) isa VariableVector, vs)
            @test all(t -> linear_index(collection_type(t[2])) === t[1], enumerate(vs))
            @test all(v -> marginal_form_constraint(v) == test_options.marginal_form_constraint, vs)
            @test all(v -> marginal_form_check_strategy(v) == test_options.marginal_form_check_strategy, vs)
            @test all(v -> messages_form_constraint(v) == test_options.messages_form_constraint, vs)
            @test all(v -> messages_form_check_strategy(v) == test_options.messages_form_check_strategy, vs)
            @test all(v -> proxy_variables(v) == test_options.proxy_variables, vs)
            @test all(v -> prod_constraint(v) == test_options.prod_constraint, vs)
            @test all(v -> prod_strategy(v) == test_options.prod_strategy, vs)
        end

        for sym in (:x, :y, :z), l in (10, 20), r in (10, 20)
            for vs in (randomvar(test_options, sym, l, r), randomvar(test_options, sym, (l, r)))
                @test israndom(vs) 
                @test size(vs) === (l, r)
                @test length(vs) === l * r
                @test vs isa Matrix
                @test all(v -> israndom(v), vs)
                @test all(v -> name(v) === sym, vs)
                @test all(v -> collection_type(v) isa VariableArray, vs)
                @test all(t -> linear_index(collection_type(t[2])) === t[1], enumerate(vs))
                @test all(v -> marginal_form_constraint(v) == test_options.marginal_form_constraint, vs)
                @test all(v -> marginal_form_check_strategy(v) == test_options.marginal_form_check_strategy, vs)
                @test all(v -> messages_form_constraint(v) == test_options.messages_form_constraint, vs)
                @test all(v -> messages_form_check_strategy(v) == test_options.messages_form_check_strategy, vs)
                @test all(v -> proxy_variables(v) == test_options.proxy_variables, vs)
                @test all(v -> prod_constraint(v) == test_options.prod_constraint, vs)
                @test all(v -> prod_strategy(v) == test_options.prod_strategy, vs)
            end
        end

    end

end

end