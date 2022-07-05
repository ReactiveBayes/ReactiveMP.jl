module ReactiveMPNodeTest

using Test
using ReactiveMP
using Rocket
using Distributions

@testset "FactorNode" begin

    @testset "Common" begin
        @test ReactiveMP.as_node_functional_form(() -> nothing) === ReactiveMP.ValidNodeFunctionalForm()
        @test ReactiveMP.as_node_functional_form(2) === ReactiveMP.UndefinedNodeFunctionalForm()

        @test isdeterministic(Deterministic()) === true
        @test isdeterministic(Deterministic) === true
        @test isdeterministic(Stochastic()) === false
        @test isdeterministic(Stochastic) === false
        @test isstochastic(Deterministic()) === false
        @test isstochastic(Deterministic) === false
        @test isstochastic(Stochastic()) === true
        @test isstochastic(Stochastic) === true

        @test sdtype(() -> nothing) === Deterministic()
        @test_throws MethodError sdtype(0)
    end

    @testset "Functional dependencies pipelines" begin 

        struct DummyStochasticNode end

        @node DummyStochasticNode Stochastic [ x, y, z ]

        function make_dummy_model(factorisation, pipeline)
            m = ReactiveMP.FactorGraphModel()
            x = randomvar(m, :x)
            y = randomvar(m, :y)
            z = randomvar(m, :z)
            make_node(m, FactorNodeCreationOptions(nothing, nothing, nothing), Uninformative, x)
            make_node(m, FactorNodeCreationOptions(nothing, nothing, nothing), Uninformative, y)
            make_node(m, FactorNodeCreationOptions(nothing, nothing, nothing), Uninformative, z)
            node = make_node(m, FactorNodeCreationOptions(factorisation, nothing, pipeline), DummyStochasticNode, x, y, z)
            activate!(m)
            return m, x, y, z, node
        end

        @testset "Default functional dependencies" begin 

            @testset "Default functional dependencies: FullFactorisation" begin 
                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), DefaultFunctionalDependencies())

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === DefaultFunctionalDependencies()

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 2 && name(x_msgdeps[1]) === :y && name(x_msgdeps[2]) === :z
                @test length(x_mgdeps) === 0

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :z
                @test length(y_mgdeps) === 0

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y
                @test length(z_mgdeps) === 0
            end

            @testset "Default functional dependencies: MeanField" begin 
                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), DefaultFunctionalDependencies())

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === DefaultFunctionalDependencies()

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 0 
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :y && name(x_mgdeps[2]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 0 
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 0 
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y
            end

            @testset "Default functional dependencies: Structured factorisation" begin 
                # We test `(x, y), (z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 2), (3, )), DefaultFunctionalDependencies())

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === DefaultFunctionalDependencies()

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :y
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :x 
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 0 
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :x_y

                ## --- ##

                # We test `(x, ), (y, z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, ), (2, 3)), DefaultFunctionalDependencies())

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === DefaultFunctionalDependencies()

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y_z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :z 
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :x

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :y
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :x

                ## --- ##

                # We test `(x, z), (y, )` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 3), (2, )), DefaultFunctionalDependencies())

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === DefaultFunctionalDependencies()

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :z
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 0
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :x_z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :x
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :y
            end

        end

        @testset "Require inbound message functional dependencies" begin 

            @testset "Require inbound message functional dependencies: FullFactorisation" begin 
                # Require inbound message on `x`
                pipeline = RequireInboundFunctionalDependencies((1, ), (NormalMeanVariance(0.123, 0.123), ))

                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 3 && name(x_msgdeps[1]) === :x && name(x_msgdeps[2]) === :y && name(x_msgdeps[3]) === :z
                @test length(x_mgdeps) === 0
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(x_msgdeps[1]))) == (0.123, 0.123)

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :z
                @test length(y_mgdeps) === 0

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y
                @test length(z_mgdeps) === 0

                ## -- ## 

                # Require inbound message on `y` and `z`
                pipeline = RequireInboundFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 2 && name(x_msgdeps[1]) === :y && name(x_msgdeps[2]) === :z
                @test length(x_mgdeps) === 0

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z
                @test length(y_mgdeps) === 0
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(y_msgdeps[2]))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 0
                @test isnothing(Rocket.getrecent(ReactiveMP.messagein(z_msgdeps[3])))
            end

            @testset "Require inbound message functional dependencies: MeanField" begin 
                # Require inbound message on `x`
                pipeline = RequireInboundFunctionalDependencies((1, ), (NormalMeanVariance(0.123, 0.123), ))

                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :x 
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :y && name(x_mgdeps[2]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(x_msgdeps[1]))) == (0.123, 0.123)

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 0
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 0
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y

                ## -- ## 

                # Require inbound message on `y` and `z`
                pipeline = RequireInboundFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :y && name(x_mgdeps[2]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :y
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(y_msgdeps[1]))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :z
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y
                @test isnothing(Rocket.getrecent(ReactiveMP.messagein(z_msgdeps[1])))
            end

            @testset "Require inbound message dependencies: Structured factorisation" begin 
                # Require inbound message on `y` and `z`
                pipeline = RequireInboundFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, y), (z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 2), (3, )), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :y
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(y_msgdeps[2]))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :z
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :x_y
                @test isnothing(Rocket.getrecent(ReactiveMP.messagein(z_msgdeps[1])))

                ## --- ##

                # Require inbound message on `y` and `z`
                pipeline = RequireInboundFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, ), (y, z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, ), (2, 3)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y_z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :y && name(y_msgdeps[2]) === :z 
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :x
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(y_msgdeps[1]))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :y && name(z_msgdeps[2]) === :z
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :x
                @test isnothing(Rocket.getrecent(ReactiveMP.messagein(z_msgdeps[2])))

                ## --- ##

                # Require inbound message on `y` and `z`
                pipeline = RequireInboundFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, z), (y, )` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 3), (2, )), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :z
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :y
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :x_z
                @test mean_var(Rocket.getrecent(ReactiveMP.messagein(y_msgdeps[1]))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :z
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :y
                @test isnothing(Rocket.getrecent(ReactiveMP.messagein(z_msgdeps[2])))
            end

        end

        @testset "Require marginal functional dependencies" begin 

            @testset "Require marginal functional dependencies: FullFactorisation" begin 
                # Require marginal on `x`
                pipeline = RequireMarginalFunctionalDependencies((1, ), (NormalMeanVariance(0.123, 0.123), ))

                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 2 && name(x_msgdeps[1]) === :y && name(x_msgdeps[2]) === :z
                @test length(x_mgdeps) === 1 &&  name(x_mgdeps[1]) === :x
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(x, IncludeAll()))) == (0.123, 0.123)

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :z
                @test length(y_mgdeps) === 0

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y
                @test length(z_mgdeps) === 0

                ## -- ## 

                # Require marginals on `y` and `z`
                pipeline = RequireMarginalFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 2 && name(x_msgdeps[1]) === :y && name(x_msgdeps[2]) === :z
                @test length(x_mgdeps) === 0

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 2 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :z
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :y
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(y, IncludeAll()))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 2 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :z
                @test isnothing(Rocket.getrecent(ReactiveMP.getmarginal(z, IncludeAll())))
            end

            @testset "Require marginal functional dependencies: MeanField" begin 
                # Require marginal on `x`
                pipeline = RequireMarginalFunctionalDependencies((1, ), (NormalMeanVariance(0.123, 0.123), ))

                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 0 
                @test length(x_mgdeps) === 3 && name(x_mgdeps[1]) === :x && name(x_mgdeps[2]) === :y && name(x_mgdeps[3]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(x, IncludeAll()))) == (0.123, 0.123)

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 0
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 0
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y

                ## -- ## 

                # Require marginals on `y` and `z`
                pipeline = RequireMarginalFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :y && name(x_mgdeps[2]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 0 
                @test length(y_mgdeps) === 3 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :y && name(y_mgdeps[3]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(y, IncludeAll()))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 0
                @test length(z_mgdeps) === 3 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y && name(z_mgdeps[3]) === :z
                @test isnothing(Rocket.getrecent(ReactiveMP.getmarginal(z, IncludeAll())))
            end

            @testset "Require marginal functional dependencies: Structured factorisation" begin 
                # Require marginal on `y` and `z`
                pipeline = RequireMarginalFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, y), (z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 2), (3, )), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :y
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :x
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :y && name(y_mgdeps[2]) === :z
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(y, IncludeAll()))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 0
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x_y && name(z_mgdeps[2]) === :z
                @test isnothing(Rocket.getrecent(ReactiveMP.getmarginal(z, IncludeAll())))

                ## --- ##

                # Require marginals on `y` and `z`
                pipeline = RequireMarginalFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, ), (y, z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, ), (2, 3)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 0
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y_z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 1 && name(y_msgdeps[1]) === :z 
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :y
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(y, IncludeAll()))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :y
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :z
                @test isnothing(Rocket.getrecent(ReactiveMP.getmarginal(z, IncludeAll())))

                ## --- ##

                # Require marginals on `y` and `z`
                pipeline = RequireMarginalFunctionalDependencies((2, 3), (NormalMeanVariance(0.123, 0.123), nothing))

                # We test `(x, z), (y, )` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 3), (2, )), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 1 && name(x_msgdeps[1]) === :z
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :y

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 0
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x_z && name(y_mgdeps[2]) === :y
                @test mean_var(Rocket.getrecent(ReactiveMP.getmarginal(y, IncludeAll()))) == (0.123, 0.123)

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 1 && name(z_msgdeps[1]) === :x
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :y && name(z_mgdeps[2]) === :z
                @test isnothing(Rocket.getrecent(ReactiveMP.getmarginal(z, IncludeAll())))
            end

        end

        @testset "Require everything functional dependencies" begin 

            @testset "Require everything functional dependencies: FullFactorisation" begin 
                pipeline = RequireEverythingFunctionalDependencies()

                # We test `FullFactorisation` case here
                m, x, y, z, node = make_dummy_model(FullFactorisation(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 3 && name(x_msgdeps[1]) === :x && name(x_msgdeps[2]) === :y && name(x_msgdeps[3]) === :z
                @test length(x_mgdeps) === 1 && name(x_mgdeps[1]) === :x_y_z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z
                @test length(y_mgdeps) === 1 && name(y_mgdeps[1]) === :x_y_z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 1 && name(z_mgdeps[1]) === :x_y_z
            end

            @testset "Require everything functional dependencies: MeanField" begin 
                pipeline = RequireEverythingFunctionalDependencies()

                # We test `MeanField` case here
                m, x, y, z, node = make_dummy_model(MeanField(), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 3 && name(x_mgdeps[1]) === :x && name(x_mgdeps[2]) === :y && name(x_mgdeps[3]) === :z 
                @test length(x_mgdeps) === 3 && name(x_mgdeps[1]) === :x && name(x_mgdeps[2]) === :y && name(x_mgdeps[3]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z 
                @test length(y_mgdeps) === 3 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :y && name(y_mgdeps[3]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 3 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y && name(z_mgdeps[3]) === :z
            end

            @testset "Require everything dependencies: Structured factorisation" begin 
                pipeline = RequireEverythingFunctionalDependencies()

                # We test `(x, y), (z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 2), (3, )), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 3 && name(x_msgdeps[1]) === :x && name(x_msgdeps[2]) === :y && name(x_msgdeps[3]) === :z
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :x_y && name(x_mgdeps[2]) === :z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x_y && name(y_mgdeps[2]) === :z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x_y && name(z_mgdeps[2]) === :z

                ## --- ##

                pipeline = RequireEverythingFunctionalDependencies()

                # We test `(x, ), (y, z)` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, ), (2, 3)), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 3 && name(x_msgdeps[1]) === :x && name(x_msgdeps[2]) === :y && name(x_msgdeps[3]) === :z
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :x && name(x_mgdeps[2]) === :y_z

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x && name(y_mgdeps[2]) === :y_z

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z) 
                
                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x && name(z_mgdeps[2]) === :y_z

                ## --- ##

                pipeline = RequireEverythingFunctionalDependencies()

                # We test `(x, z), (y, )` factorisation case here
                m, x, y, z, node = make_dummy_model(((1, 3), (2, )), pipeline)

                # Test that pipeline dependencies have been set properly
                @test ReactiveMP.get_pipeline_dependencies(ReactiveMP.getpipeline(node)) === pipeline

                x_msgdeps, x_mgdeps = ReactiveMP.functional_dependencies(node, :x) 

                @test length(x_msgdeps) === 3 && name(x_msgdeps[1]) === :x && name(x_msgdeps[2]) === :y && name(x_msgdeps[3]) === :z
                @test length(x_mgdeps) === 2 && name(x_mgdeps[1]) === :x_z && name(x_mgdeps[2]) === :y

                y_msgdeps, y_mgdeps = ReactiveMP.functional_dependencies(node, :y) 
                
                @test length(y_msgdeps) === 3 && name(y_msgdeps[1]) === :x && name(y_msgdeps[2]) === :y && name(y_msgdeps[3]) === :z
                @test length(y_mgdeps) === 2 && name(y_mgdeps[1]) === :x_z && name(y_mgdeps[2]) === :y

                z_msgdeps, z_mgdeps = ReactiveMP.functional_dependencies(node, :z)
                
                @test length(z_msgdeps) === 3 && name(z_msgdeps[1]) === :x && name(z_msgdeps[2]) === :y && name(z_msgdeps[3]) === :z
                @test length(z_mgdeps) === 2 && name(z_mgdeps[1]) === :x_z && name(z_mgdeps[2]) === :y
            end

        end

    end

    @testset "@node macro" begin

        # Testing Stochastic node specification

        struct CustomStochasticNode end

        @node CustomStochasticNode Stochastic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

        @test ReactiveMP.interface_get_index(Val{:CustomStochasticNode}, Val{:out}) === 1
        @test ReactiveMP.interface_get_index(Val{:CustomStochasticNode}, Val{:x}) === 2
        @test ReactiveMP.interface_get_index(Val{:CustomStochasticNode}, Val{:y}) === 3
        @test ReactiveMP.interface_get_index(Val{:CustomStochasticNode}, Val{:z}) === 4

        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:out}) === :out
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:x}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:y}) === :y
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:z}) === :z

        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:xx}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{:yy}) === :y

        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{1}) === :out
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{2}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{3}) === :y
        @test ReactiveMP.interface_get_name(Val{:CustomStochasticNode}, Val{4}) === :z

        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1,), (2,), (3,), (4,))) ===
              ((1,), (2,), (3,), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2), (3,), (4,))) === ((1, 2), (3,), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2, 3), (4,))) === ((1, 2, 3), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2, 3), (4,))) === ((1, 2, 3), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, ((1, 2, 3, 4),)) === ((1, 2, 3, 4),)

        @test ReactiveMP.collect_factorisation(CustomStochasticNode, MeanField()) === ((1,), (2,), (3,), (4,))
        @test ReactiveMP.collect_factorisation(CustomStochasticNode, FullFactorisation()) === ((1, 2, 3, 4),)

        @test sdtype(CustomStochasticNode) === Stochastic()

        cx = constvar(:cx, 1.0)
        cy = constvar(:cy, 1.0)
        cz = constvar(:cz, 1.0)

        model = FactorGraphModel()

        snode, svar = make_node(
            model,
            FactorNodeCreationOptions(MeanField(), nothing, nothing),
            CustomStochasticNode,
            AutoVar(:cout),
            cx,
            cy,
            cz
        )

        @test snode ∈ getnodes(model)
        @test svar ∈ getrandom(model)

        @test snode !== nothing
        @test typeof(svar) <: RandomVariable
        @test factorisation(snode) === ((1,), (2,), (3,), (4,))

        # Testing Deterministic node specification

        struct CustomDeterministicNode end

        CustomDeterministicNode(x, y, z) = x + y + z

        @node CustomDeterministicNode Deterministic [out, (x, aliases = [xx]), (y, aliases = [yy]), z]

        @test ReactiveMP.interface_get_index(Val{:CustomDeterministicNode}, Val{:out}) === 1
        @test ReactiveMP.interface_get_index(Val{:CustomDeterministicNode}, Val{:x}) === 2
        @test ReactiveMP.interface_get_index(Val{:CustomDeterministicNode}, Val{:y}) === 3
        @test ReactiveMP.interface_get_index(Val{:CustomDeterministicNode}, Val{:z}) === 4

        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:out}) === :out
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:x}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:y}) === :y
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:z}) === :z

        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:xx}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{:yy}) === :y

        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{1}) === :out
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{2}) === :x
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{3}) === :y
        @test ReactiveMP.interface_get_name(Val{:CustomDeterministicNode}, Val{4}) === :z

        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, ((1,), (2,), (3,), (4,))) === ((1, 2, 3, 4),)
        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, ((1, 2), (3,), (4,))) === ((1, 2, 3, 4),)
        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, ((1, 2, 3), (4,))) === ((1, 2, 3, 4),)
        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, ((1, 2, 3), (4,))) === ((1, 2, 3, 4),)
        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, ((1, 2, 3, 4),)) === ((1, 2, 3, 4),)

        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, MeanField()) === ((1, 2, 3, 4),)
        @test ReactiveMP.collect_factorisation(CustomDeterministicNode, FullFactorisation()) === ((1, 2, 3, 4),)

        @test sdtype(CustomDeterministicNode) === Deterministic()

        cx = constvar(:cx, 1.0)
        cy = constvar(:cy, 1.0)
        cz = constvar(:cz, 1.0)

        model = FactorGraphModel()

        snode, svar = make_node(
            model,
            FactorNodeCreationOptions(MeanField(), nothing, nothing),
            CustomDeterministicNode,
            AutoVar(:cout),
            cx,
            cy,
            cz
        )

        @test svar ∈ getconstant(model)

        @test snode === nothing
        @test typeof(svar) <: ConstVariable

        # Check that same variables are not allowed

        struct DummyNodeCheckUniqueness end

        @node DummyNodeCheckUniqueness Stochastic [a, b, c]

        sx = randomvar(:rx)
        sd = datavar(:rd, Float64)
        sc = constvar(:sc, 1.0)

        vs = (sx, sd, sc)

        for a in vs, b in vs, c in vs
            input = (a, b, c)
            if length(input) != length(Set(input))
                @test_throws ErrorException make_node(DummyNodeCheckUniqueness, FactorNodeCreationOptions(), a, b, c)
            end
        end

        # Testing expected exceptions

        struct DummyStruct end

        @test_throws Exception eval(:(@node DummyStruct NotStochasticAndNotDeterministic [out, in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [1, in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [(1, aliases = [out]), in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [(out, aliases = [out]), in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic [(out, aliases = [1]), in, x]))
        @test_throws Exception eval(:(@node DummyStruct Stochastic []))
    end

    @testset "sdtype of an arbitrary distribution is Stochastic" begin
        struct DummyDistribution <: Distribution{Univariate, Continuous} end

        @test sdtype(DummyDistribution) === Stochastic()
    end

    @testset "make_node throws on Unknown distribution type" begin
        struct DummyDistribution <: Distribution{Univariate, Continuous} end

        @test_throws ErrorException ReactiveMP.make_node(
            FactorGraphModel(),
            FactorNodeCreationOptions(),
            DummyDistribution,
            AutoVar(:θ)
        )
        @test_throws ErrorException ReactiveMP.make_node(
            FactorGraphModel(),
            FactorNodeCreationOptions(),
            DummyDistribution,
            randomvar(:θ)
        )
    end
end

end
