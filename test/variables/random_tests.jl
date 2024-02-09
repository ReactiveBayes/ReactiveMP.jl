
@testitem "RandomVariable" begin
    using ReactiveMP, Rocket, BayesBase

    import ReactiveMP: UnspecifiedFormConstraint
    import ReactiveMP: collection_type, VariableIndividual, VariableVector, VariableArray, linear_index
    import ReactiveMP: prod_constraint, prod_strategy
    import ReactiveMP: proxy_variables, israndom, isproxy
    import ReactiveMP: marginal_form_constraint, marginal_form_check_strategy
    import ReactiveMP: messages_form_constraint, messages_form_check_strategy

    @testset "Simple creation" begin
        for sym in (:x, :y, :z)
            v = randomvar(sym)

            @test israndom(v)
            @test name(v) === sym
            @test collection_type(v) isa VariableIndividual
            @test marginal_form_constraint(v) isa UnspecifiedFormConstraint
            @test messages_form_constraint(v) isa UnspecifiedFormConstraint
            @test proxy_variables(v) === nothing
            @test prod_constraint(v) isa GenericProd
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
            @test all(v -> prod_constraint(v) isa GenericProd, vs)
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
                @test all(v -> prod_constraint(v) isa GenericProd, vs)
                @test all(v -> prod_strategy(v) isa FoldLeftProdStrategy, vs)
                @test !isproxy(vs)
                @test all(v -> !isproxy(v), vs)
            end
        end
    end

    @testset "Creation via options" begin
        struct CustomFunctionalFormConstraint1 <: ReactiveMP.AbstractFormConstraint end
        struct CustomFunctionalFormConstraint2 <: ReactiveMP.AbstractFormConstraint end

        test_var     = randomvar(:tmp)
        test_options = RandomVariableCreationOptions(LoggerPipelineStage(), (test_var,), ClosedProd(), FoldRightProdStrategy(), CustomFunctionalFormConstraint1(), FormConstraintCheckEach(), CustomFunctionalFormConstraint2(), FormConstraintCheckLast())

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

    @testset "Options setters" begin
        for pipeline in (EmptyPipelineStage(), LoggerPipelineStage(), DiscontinuePipelineStage())
            @test ReactiveMP.get_pipeline_stages(randomvar(ReactiveMP.randomvar_options_set_pipeline(pipeline), :x)) === pipeline
            let options = RandomVariableCreationOptions()
                @test ReactiveMP.get_pipeline_stages(randomvar(ReactiveMP.randomvar_options_set_pipeline(options, pipeline), :x)) === pipeline
            end
        end

        # here and later on we use some dummy values
        dummy = (1.0, 1, "dummy")

        for proxy_variables in dummy
            @test ReactiveMP.proxy_variables(randomvar(ReactiveMP.randomvar_options_set_proxy_variables(proxy_variables), :x)) === proxy_variables
            let options = RandomVariableCreationOptions()
                @test ReactiveMP.proxy_variables(randomvar(ReactiveMP.randomvar_options_set_proxy_variables(options, proxy_variables), :x)) === proxy_variables
            end
        end

        for prod_constraint in dummy
            @test ReactiveMP.prod_constraint(randomvar(ReactiveMP.randomvar_options_set_prod_constraint(prod_constraint), :x)) === prod_constraint
            let options = RandomVariableCreationOptions()
                @test ReactiveMP.prod_constraint(randomvar(ReactiveMP.randomvar_options_set_prod_constraint(options, prod_constraint), :x)) === prod_constraint
            end
        end

        for prod_strategy in dummy
            @test ReactiveMP.prod_strategy(randomvar(ReactiveMP.randomvar_options_set_prod_strategy(prod_strategy), :x)) === prod_strategy
            let options = RandomVariableCreationOptions()
                @test ReactiveMP.prod_strategy(randomvar(ReactiveMP.randomvar_options_set_prod_strategy(options, prod_strategy), :x)) === prod_strategy
            end
        end

        for marginal_form_constraint in dummy
            @test ReactiveMP.marginal_form_constraint(randomvar(ReactiveMP.randomvar_options_set_marginal_form_constraint(marginal_form_constraint), :x)) ===
                marginal_form_constraint
            let options = RandomVariableCreationOptions()
                @test ReactiveMP.marginal_form_constraint(randomvar(ReactiveMP.randomvar_options_set_marginal_form_constraint(options, marginal_form_constraint), :x)) ===
                    marginal_form_constraint
            end
        end

        for marginal_form_check_strategy in dummy
            @test ReactiveMP.marginal_form_check_strategy(randomvar(ReactiveMP.randomvar_options_set_marginal_form_check_strategy(marginal_form_check_strategy), :x)) ===
                marginal_form_check_strategy
            let options = RandomVariableCreationOptions()
                @test ReactiveMP.marginal_form_check_strategy(
                    randomvar(ReactiveMP.randomvar_options_set_marginal_form_check_strategy(options, marginal_form_check_strategy), :x)
                ) === marginal_form_check_strategy
            end
        end

        for messages_form_constraint in dummy
            @test ReactiveMP.messages_form_constraint(randomvar(ReactiveMP.randomvar_options_set_messages_form_constraint(messages_form_constraint), :x)) ===
                messages_form_constraint
            let options = RandomVariableCreationOptions()
                @test ReactiveMP.messages_form_constraint(randomvar(ReactiveMP.randomvar_options_set_messages_form_constraint(options, messages_form_constraint), :x)) ===
                    messages_form_constraint
            end
        end

        for messages_form_check_strategy in dummy
            @test ReactiveMP.messages_form_check_strategy(randomvar(ReactiveMP.randomvar_options_set_messages_form_check_strategy(messages_form_check_strategy), :x)) ===
                messages_form_check_strategy
            let options = RandomVariableCreationOptions()
                @test ReactiveMP.messages_form_check_strategy(
                    randomvar(ReactiveMP.randomvar_options_set_messages_form_check_strategy(options, messages_form_check_strategy), :x)
                ) === messages_form_check_strategy
            end
        end
    end

    @testset "Proxy creation" begin
        proxy_var1 = randomvar(:proxy1)
        proxy_var2 = randomvar(:proxy2)

        for sym in (:x, :y, :z)
            v1 = randomvar(ReactiveMP.randomvar_options_set_proxy_variables((proxy_var1,)), sym)
            @test israndom(v1)
            @test name(v1) === sym
            @test proxy_variables(v1) === (proxy_var1,)
            @test isproxy(v1)

            v2 = randomvar(ReactiveMP.randomvar_options_set_proxy_variables((proxy_var1, proxy_var2)), sym)
            @test israndom(v2)
            @test name(v2) === sym
            @test proxy_variables(v2) === (proxy_var1, proxy_var2)
            @test isproxy(v2)
        end
    end

    @testset "Error indexing" begin
        # This test may be removed if we implement this feature in the future
        @test_throws ErrorException randomvar(:x)[1]
    end
end
