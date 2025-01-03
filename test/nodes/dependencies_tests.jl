@testitem "collect_latest_messages" begin
    include("../testutilities.jl")
    using BayesBase

    import ReactiveMP: NodeInterface, collect_latest_messages, getdata, getrecent, default_functional_dependencies

    struct ArbitraryNode end

    @node ArbitraryNode Stochastic [a, b, c]

    a_v = ConstVariable(1)
    b_v = ConstVariable(2)
    c_v = ConstVariable(3)

    node = factornode(ArbitraryNode, [(:a, a_v), (:b, b_v), (:c, c_v)], ((1, 2, 3),))
    dependencies = default_functional_dependencies(ArbitraryNode)

    a, b, c = getinterfaces(node)

    @testset let (tag, stream) = collect_latest_messages(dependencies, node, ())
        @test tag === nothing
        @test check_stream_updated_once(stream) === nothing
    end

    @testset let (tag, stream) = collect_latest_messages(dependencies, node, (a, b, c))
        @test tag === Val((:a, :b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_messages(dependencies, node, (a, b))
        @test tag === Val((:a, :b))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2))
    end

    @testset let (tag, stream) = collect_latest_messages(dependencies, node, (b, c))
        @test tag === Val((:b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_messages(dependencies, node, (a, c))
        @test tag === Val((:a, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(3))
    end
end

@testitem "collect_latest_marginals" begin
    include("../testutilities.jl")
    using BayesBase

    import ReactiveMP:
        NodeInterface, FactorNodeLocalMarginal, getmarginal, collect_latest_marginals, getdata, getrecent, default_functional_dependencies, getlocalclusters, getmarginals

    struct ArbitraryNode end

    @node ArbitraryNode Stochastic [a, b, c]

    a_v = ConstVariable(1)
    b_v = ConstVariable(2)
    c_v = ConstVariable(3)

    node = factornode(ArbitraryNode, [(:a, a_v), (:b, b_v), (:c, c_v)], ((1,), (2,), (3,)))
    dependencies = default_functional_dependencies(ArbitraryNode)

    a, b, c = getmarginals(getlocalclusters(node))

    setmarginal!(a, getmarginal(a_v, IncludeAll()))
    setmarginal!(b, getmarginal(b_v, IncludeAll()))
    setmarginal!(c, getmarginal(c_v, IncludeAll()))

    @testset let (tag, stream) = collect_latest_marginals(dependencies, node, (a, b, c))
        @test tag === Val((:a, :b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_marginals(dependencies, node, (a, b))
        @test tag === Val((:a, :b))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2))
    end

    @testset let (tag, stream) = collect_latest_marginals(dependencies, node, (b, c))
        @test tag === Val((:b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_marginals(dependencies, node, (a, c))
        @test tag === Val((:a, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(3))
    end
end

@testitem "Various functional dependencies" begin
    include("../testutilities.jl")

    import ReactiveMP: activate!, RandomVariableActivationOptions, functional_dependencies, getinterfaces, messagein, getdata

    struct ArbitraryFactorNode end

    @node ArbitraryFactorNode Stochastic [a, b, c]

    @testset "Bethe factorization" begin
        a = randomvar()
        b = randomvar()
        c = randomvar()

        node = factornode(ArbitraryFactorNode, [(:a, a), (:b, b), (:c, c)], ((1, 2, 3),))
        interfaces = getinterfaces(node)

        activate!(a, RandomVariableActivationOptions())
        activate!(b, RandomVariableActivationOptions())
        activate!(c, RandomVariableActivationOptions())

        @testset "DefaultFunctionalDependencies" begin
            import ReactiveMP: DefaultFunctionalDependencies

            dependencies = DefaultFunctionalDependencies()

            msg_dependencies_for_a, marginal_dependencies_for_a = functional_dependencies(dependencies, node, interfaces[1], 1)
            msg_dependencies_for_b, marginal_dependencies_for_b = functional_dependencies(dependencies, node, interfaces[2], 2)
            msg_dependencies_for_c, marginal_dependencies_for_c = functional_dependencies(dependencies, node, interfaces[3], 3)

            @test interfaces[1] ∉ msg_dependencies_for_a
            @test interfaces[2] ∈ msg_dependencies_for_a
            @test interfaces[3] ∈ msg_dependencies_for_a
            @test isempty(marginal_dependencies_for_a)

            @test interfaces[1] ∈ msg_dependencies_for_b
            @test interfaces[2] ∉ msg_dependencies_for_b
            @test interfaces[3] ∈ msg_dependencies_for_b
            @test isempty(marginal_dependencies_for_b)

            @test interfaces[1] ∈ msg_dependencies_for_c
            @test interfaces[2] ∈ msg_dependencies_for_c
            @test interfaces[3] ∉ msg_dependencies_for_c
            @test isempty(marginal_dependencies_for_c)

            @test check_stream_not_updated(messagein(interfaces[1]))
            @test check_stream_not_updated(messagein(interfaces[2]))
            @test check_stream_not_updated(messagein(interfaces[3]))
        end

        @testset "RequireMessageFunctionalDependencies(a = nothing)" begin
            import ReactiveMP: RequireMessageFunctionalDependencies

            dependencies = RequireMessageFunctionalDependencies(a = nothing)

            msg_dependencies_for_a, marginal_dependencies_for_a = functional_dependencies(dependencies, node, interfaces[1], 1)
            msg_dependencies_for_b, marginal_dependencies_for_b = functional_dependencies(dependencies, node, interfaces[2], 2)
            msg_dependencies_for_c, marginal_dependencies_for_c = functional_dependencies(dependencies, node, interfaces[3], 3)

            @test interfaces[1] ∈ msg_dependencies_for_a
            @test interfaces[2] ∈ msg_dependencies_for_a
            @test interfaces[3] ∈ msg_dependencies_for_a
            @test isempty(marginal_dependencies_for_a)

            @test interfaces[1] ∈ msg_dependencies_for_b
            @test interfaces[2] ∉ msg_dependencies_for_b
            @test interfaces[3] ∈ msg_dependencies_for_b
            @test isempty(marginal_dependencies_for_b)

            @test interfaces[1] ∈ msg_dependencies_for_c
            @test interfaces[2] ∈ msg_dependencies_for_c
            @test interfaces[3] ∉ msg_dependencies_for_c
            @test isempty(marginal_dependencies_for_c)

            @test check_stream_not_updated(messagein(interfaces[1]))
            @test check_stream_not_updated(messagein(interfaces[2]))
            @test check_stream_not_updated(messagein(interfaces[3]))
        end

        @testset "RequireMessageFunctionalDependencies(b = ...)" begin
            import ReactiveMP: RequireMessageFunctionalDependencies

            for initialmessage in (1, 2.0, "hello")
                dependencies = RequireMessageFunctionalDependencies(b = initialmessage)

                msg_dependencies_for_a, marginal_dependencies_for_a = functional_dependencies(dependencies, node, interfaces[1], 1)
                msg_dependencies_for_b, marginal_dependencies_for_b = functional_dependencies(dependencies, node, interfaces[2], 2)
                msg_dependencies_for_c, marginal_dependencies_for_c = functional_dependencies(dependencies, node, interfaces[3], 3)

                @test interfaces[1] ∉ msg_dependencies_for_a
                @test interfaces[2] ∈ msg_dependencies_for_a
                @test interfaces[3] ∈ msg_dependencies_for_a
                @test isempty(marginal_dependencies_for_a)

                @test interfaces[1] ∈ msg_dependencies_for_b
                @test interfaces[2] ∈ msg_dependencies_for_b
                @test interfaces[3] ∈ msg_dependencies_for_b
                @test isempty(marginal_dependencies_for_b)

                @test interfaces[1] ∈ msg_dependencies_for_c
                @test interfaces[2] ∈ msg_dependencies_for_c
                @test interfaces[3] ∉ msg_dependencies_for_c
                @test isempty(marginal_dependencies_for_c)

                @test check_stream_not_updated(messagein(interfaces[1]))
                @test getdata(check_stream_updated_once(messagein(interfaces[2]))) === initialmessage
                @test check_stream_not_updated(messagein(interfaces[3]))
            end
        end

        @testset "RequireMarginalFunctionalDependencies(a = nothing)" begin
            import ReactiveMP: RequireMarginalFunctionalDependencies

            dependencies = RequireMarginalFunctionalDependencies(a = nothing)

            msg_dependencies_for_a, marginal_dependencies_for_a = functional_dependencies(dependencies, node, interfaces[1], 1)
            msg_dependencies_for_b, marginal_dependencies_for_b = functional_dependencies(dependencies, node, interfaces[2], 2)
            msg_dependencies_for_c, marginal_dependencies_for_c = functional_dependencies(dependencies, node, interfaces[3], 3)

            @test interfaces[1] ∉ msg_dependencies_for_a
            @test interfaces[2] ∈ msg_dependencies_for_a
            @test interfaces[3] ∈ msg_dependencies_for_a
            @test !isempty(marginal_dependencies_for_a)
            @test length(marginal_dependencies_for_a) === 1
            @test check_stream_not_updated(getmarginal(first(marginal_dependencies_for_a)))

            @test interfaces[1] ∈ msg_dependencies_for_b
            @test interfaces[2] ∉ msg_dependencies_for_b
            @test interfaces[3] ∈ msg_dependencies_for_b
            @test isempty(marginal_dependencies_for_b)

            @test interfaces[1] ∈ msg_dependencies_for_c
            @test interfaces[2] ∈ msg_dependencies_for_c
            @test interfaces[3] ∉ msg_dependencies_for_c
            @test isempty(marginal_dependencies_for_c)
        end

        @testset "RequireMarginalFunctionalDependencies(a = ...)" begin
            import ReactiveMP: RequireMessageFunctionalDependencies

            for initialmarginal in (1, 2.0, "hello")
                import ReactiveMP: RequireMarginalFunctionalDependencies

                dependencies = RequireMarginalFunctionalDependencies(a = initialmarginal)

                msg_dependencies_for_a, marginal_dependencies_for_a = functional_dependencies(dependencies, node, interfaces[1], 1)
                msg_dependencies_for_b, marginal_dependencies_for_b = functional_dependencies(dependencies, node, interfaces[2], 2)
                msg_dependencies_for_c, marginal_dependencies_for_c = functional_dependencies(dependencies, node, interfaces[3], 3)

                @test interfaces[1] ∉ msg_dependencies_for_a
                @test interfaces[2] ∈ msg_dependencies_for_a
                @test interfaces[3] ∈ msg_dependencies_for_a
                @test !isempty(marginal_dependencies_for_a)
                @test length(marginal_dependencies_for_a) === 1
                @test getdata(check_stream_updated_once(getmarginal(first(marginal_dependencies_for_a)))) === initialmarginal

                @test interfaces[1] ∈ msg_dependencies_for_b
                @test interfaces[2] ∉ msg_dependencies_for_b
                @test interfaces[3] ∈ msg_dependencies_for_b
                @test isempty(marginal_dependencies_for_b)

                @test interfaces[1] ∈ msg_dependencies_for_c
                @test interfaces[2] ∈ msg_dependencies_for_c
                @test interfaces[3] ∉ msg_dependencies_for_c
                @test isempty(marginal_dependencies_for_c)
            end
        end
    end

    @testset "MeanField factorization" begin
        import ReactiveMP: name

        a = randomvar()
        b = randomvar()
        c = randomvar()

        node = factornode(ArbitraryFactorNode, [(:a, a), (:b, b), (:c, c)], ((1,), (2,), (3,)))
        interfaces = getinterfaces(node)

        activate!(a, RandomVariableActivationOptions())
        activate!(b, RandomVariableActivationOptions())
        activate!(c, RandomVariableActivationOptions())

        @testset "DefaultFunctionalDependencies" begin
            import ReactiveMP: DefaultFunctionalDependencies

            dependencies = DefaultFunctionalDependencies()

            msg_dependencies_for_a, marginal_dependencies_for_a = functional_dependencies(dependencies, node, interfaces[1], 1)
            msg_dependencies_for_b, marginal_dependencies_for_b = functional_dependencies(dependencies, node, interfaces[2], 2)
            msg_dependencies_for_c, marginal_dependencies_for_c = functional_dependencies(dependencies, node, interfaces[3], 3)

            @test :a ∉ name.(marginal_dependencies_for_a)
            @test :b ∈ name.(marginal_dependencies_for_a)
            @test :c ∈ name.(marginal_dependencies_for_a)
            @test isempty(msg_dependencies_for_a)

            @test :a ∈ name.(marginal_dependencies_for_b)
            @test :b ∉ name.(marginal_dependencies_for_b)
            @test :c ∈ name.(marginal_dependencies_for_b)
            @test isempty(msg_dependencies_for_b)

            @test :a ∈ name.(marginal_dependencies_for_c)
            @test :b ∈ name.(marginal_dependencies_for_c)
            @test :c ∉ name.(marginal_dependencies_for_c)
            @test isempty(msg_dependencies_for_c)
        end

        @testset "RequireMessageFunctionalDependencies(a = nothing)" begin
            import ReactiveMP: RequireMessageFunctionalDependencies

            dependencies = RequireMessageFunctionalDependencies(a = nothing)

            msg_dependencies_for_a, marginal_dependencies_for_a = functional_dependencies(dependencies, node, interfaces[1], 1)
            msg_dependencies_for_b, marginal_dependencies_for_b = functional_dependencies(dependencies, node, interfaces[2], 2)
            msg_dependencies_for_c, marginal_dependencies_for_c = functional_dependencies(dependencies, node, interfaces[3], 3)

            @test :a ∉ name.(marginal_dependencies_for_a)
            @test :b ∈ name.(marginal_dependencies_for_a)
            @test :c ∈ name.(marginal_dependencies_for_a)
            @test interfaces[1] ∈ msg_dependencies_for_a
            @test !isempty(msg_dependencies_for_a)

            @test :a ∈ name.(marginal_dependencies_for_b)
            @test :b ∉ name.(marginal_dependencies_for_b)
            @test :c ∈ name.(marginal_dependencies_for_b)
            @test isempty(msg_dependencies_for_b)

            @test :a ∈ name.(marginal_dependencies_for_c)
            @test :b ∈ name.(marginal_dependencies_for_c)
            @test :c ∉ name.(marginal_dependencies_for_c)
            @test isempty(msg_dependencies_for_c)

            @test check_stream_not_updated(messagein(interfaces[1]))
            @test check_stream_not_updated(messagein(interfaces[2]))
            @test check_stream_not_updated(messagein(interfaces[3]))
        end

        @testset "RequireMessageFunctionalDependencies(b = ...)" begin
            import ReactiveMP: RequireMessageFunctionalDependencies

            for initialmessage in (1, 2.0, "hello")
                dependencies = RequireMessageFunctionalDependencies(b = initialmessage)

                msg_dependencies_for_a, marginal_dependencies_for_a = functional_dependencies(dependencies, node, interfaces[1], 1)
                msg_dependencies_for_b, marginal_dependencies_for_b = functional_dependencies(dependencies, node, interfaces[2], 2)
                msg_dependencies_for_c, marginal_dependencies_for_c = functional_dependencies(dependencies, node, interfaces[3], 3)

                @test :a ∉ name.(marginal_dependencies_for_a)
                @test :b ∈ name.(marginal_dependencies_for_a)
                @test :c ∈ name.(marginal_dependencies_for_a)
                @test isempty(msg_dependencies_for_a)

                @test :a ∈ name.(marginal_dependencies_for_b)
                @test :b ∉ name.(marginal_dependencies_for_b)
                @test :c ∈ name.(marginal_dependencies_for_b)
                @test interfaces[2] ∈ msg_dependencies_for_b
                @test !isempty(msg_dependencies_for_b)

                @test :a ∈ name.(marginal_dependencies_for_c)
                @test :b ∈ name.(marginal_dependencies_for_c)
                @test :c ∉ name.(marginal_dependencies_for_c)
                @test isempty(msg_dependencies_for_c)

                @test check_stream_not_updated(messagein(interfaces[1]))
                @test getdata(check_stream_updated_once(messagein(interfaces[2]))) === initialmessage
                @test check_stream_not_updated(messagein(interfaces[3]))
            end
        end

        @testset "RequireMarginalFunctionalDependencies(a = nothing)" begin
            import ReactiveMP: RequireMarginalFunctionalDependencies

            dependencies = RequireMarginalFunctionalDependencies(a = nothing)

            msg_dependencies_for_a, marginal_dependencies_for_a = functional_dependencies(dependencies, node, interfaces[1], 1)
            msg_dependencies_for_b, marginal_dependencies_for_b = functional_dependencies(dependencies, node, interfaces[2], 2)
            msg_dependencies_for_c, marginal_dependencies_for_c = functional_dependencies(dependencies, node, interfaces[3], 3)

            @test :a ∈ name.(marginal_dependencies_for_a)
            @test :b ∈ name.(marginal_dependencies_for_a)
            @test :c ∈ name.(marginal_dependencies_for_a)
            @test isempty(msg_dependencies_for_a)
            @test check_stream_not_updated(getmarginal(first(marginal_dependencies_for_a)))

            @test :a ∈ name.(marginal_dependencies_for_b)
            @test :b ∉ name.(marginal_dependencies_for_b)
            @test :c ∈ name.(marginal_dependencies_for_b)
            @test isempty(msg_dependencies_for_b)

            @test :a ∈ name.(marginal_dependencies_for_c)
            @test :b ∈ name.(marginal_dependencies_for_c)
            @test :c ∉ name.(marginal_dependencies_for_c)
            @test isempty(msg_dependencies_for_c)
        end

        @testset "RequireMarginalFunctionalDependencies(a = ...)" begin
            import ReactiveMP: RequireMessageFunctionalDependencies

            for initialmarginal in (1, 2.0, "hello")
                import ReactiveMP: RequireMarginalFunctionalDependencies

                dependencies = RequireMarginalFunctionalDependencies(a = initialmarginal)

                msg_dependencies_for_a, marginal_dependencies_for_a = functional_dependencies(dependencies, node, interfaces[1], 1)
                msg_dependencies_for_b, marginal_dependencies_for_b = functional_dependencies(dependencies, node, interfaces[2], 2)
                msg_dependencies_for_c, marginal_dependencies_for_c = functional_dependencies(dependencies, node, interfaces[3], 3)

                @test :a ∈ name.(marginal_dependencies_for_a)
                @test :b ∈ name.(marginal_dependencies_for_a)
                @test :c ∈ name.(marginal_dependencies_for_a)
                @test isempty(msg_dependencies_for_a)
                @test getdata(check_stream_updated_once(getmarginal(first(marginal_dependencies_for_a)))) === initialmarginal

                @test :a ∈ name.(marginal_dependencies_for_b)
                @test :b ∉ name.(marginal_dependencies_for_b)
                @test :c ∈ name.(marginal_dependencies_for_b)
                @test isempty(msg_dependencies_for_b)

                @test :a ∈ name.(marginal_dependencies_for_c)
                @test :b ∈ name.(marginal_dependencies_for_c)
                @test :c ∉ name.(marginal_dependencies_for_c)
                @test isempty(msg_dependencies_for_c)
            end
        end
    end
end

@testitem "Functional dependencies may change depending on the metadata from options" begin
    # This test demonstrates how functional dependencies can be customized based on metadata
    # passed during node activation. This is useful when:
    # 1. The same node type needs different message passing behaviors in different contexts
    # 2. Users want to override the default message passing behavior without creating a new node type
    #
    # The test creates a custom node with 3 interfaces (out, in1, in2) and shows how:
    # - With meta = :use_a, the output message depends only on in1
    # - With meta = :use_b, the output message depends only on in2
    # - With meta = nothing, it falls back to default dependencies
    #
    # This pattern allows for flexible message passing schemes that can be configured at runtime
    # rather than being hardcoded into the node type.

    include("../testutilities.jl")
    using BayesBase, Rocket

    import ReactiveMP: NodeInterface, collect_functional_dependencies, getdata, getrecent, activate!, getmetadata, name, getinterface
    import ReactiveMP: FunctionalDependencies, functional_dependencies, DefaultFunctionalDependencies, FactorNodeActivationOptions, getdependecies

    # Define a custom node for testing
    struct CustomMetaNode end

    @node CustomMetaNode Stochastic [out, in1, in2]

    # Define custom functional dependencies that we'll use based on meta
    struct CustomDependencyA <: FunctionalDependencies end
    struct CustomDependencyB <: FunctionalDependencies end

    # Define how meta affects functional dependencies
    ReactiveMP.collect_functional_dependencies(::Type{CustomMetaNode}, options::FactorNodeActivationOptions) =
        ReactiveMP.collect_functional_dependencies(CustomMetaNode, options, getmetadata(options))

    # Mock different behavior for our custom dependencies
    ReactiveMP.collect_functional_dependencies(::Type{CustomMetaNode}, ::FactorNodeActivationOptions, meta::Symbol) = meta === :use_a ? CustomDependencyA() : CustomDependencyB()
    ReactiveMP.collect_functional_dependencies(::Type{CustomMetaNode}, options::FactorNodeActivationOptions, meta::Nothing) =
        ReactiveMP.collect_functional_dependencies(CustomMetaNode, getdependecies(options))

    # Mock different behavior for our custom dependencies
    function ReactiveMP.functional_dependencies(::CustomDependencyA, factornode, interface, iindex)
        # CustomDependencyA only depends on in1
        msg_deps = name(interface) === :out ? (getinterface(factornode, 2),) : () # only in1
        return (msg_deps, ())
    end

    function ReactiveMP.functional_dependencies(::CustomDependencyB, factornode, interface, iindex)
        # CustomDependencyB only depends on in2
        msg_deps = name(interface) === :out ? (getinterface(factornode, 3),) : () # only in2
        return (msg_deps, ())
    end

    # Create test variables
    out_v = randomvar()
    in1_v = ConstVariable(1.0)
    in2_v = ConstVariable(2.0)

    @testset "use_a metadata results in CustomDependencyA" begin
        options_a = FactorNodeActivationOptions(:use_a, nothing, nothing, nothing, AsapScheduler(), nothing)
        deps = collect_functional_dependencies(CustomMetaNode, options_a)
        @test deps isa CustomDependencyA
    end

    @testset "use_b metadata results in CustomDependencyB" begin
        options_b = FactorNodeActivationOptions(:use_b, nothing, nothing, nothing, AsapScheduler(), nothing)
        deps = collect_functional_dependencies(CustomMetaNode, options_b)
        @test deps isa CustomDependencyB
    end

    @testset "no metadata falls back to default dependencies" begin
        options_default = FactorNodeActivationOptions(nothing, nothing, nothing, nothing, AsapScheduler(), nothing)
        deps = collect_functional_dependencies(CustomMetaNode, options_default)
        @test deps isa DefaultFunctionalDependencies
    end

    @testset "Dependencies change based on meta" begin
        # Create node with meta :use_a
        node_a = factornode(CustomMetaNode, [(:out, out_v), (:in1, in1_v), (:in2, in2_v)], ((1,),))
        options_a = FactorNodeActivationOptions(:use_a, nothing, nothing, nothing, AsapScheduler(), nothing)
        deps_a = collect_functional_dependencies(CustomMetaNode, options_a)
        activate!(node_a, options_a)

        out_interface_a = getinterface(node_a, 1)
        msg_deps_a, marg_deps_a = functional_dependencies(deps_a, node_a, out_interface_a, 1)
        @test length(msg_deps_a) == 1
        @test name(first(msg_deps_a)) === :in1

        # Test that functional dependencies are different with meta :use_b
        node_b = factornode(CustomMetaNode, [(:out, out_v), (:in1, in1_v), (:in2, in2_v)], ((1,),))
        options_b = FactorNodeActivationOptions(:use_b, nothing, nothing, nothing, AsapScheduler(), nothing)
        deps_b = collect_functional_dependencies(CustomMetaNode, options_b)
        activate!(node_b, options_b)
        out_interface_b = getinterface(node_b, 1)
        msg_deps_b, marg_deps_b = functional_dependencies(deps_b, node_b, out_interface_b, 1)
        @test length(msg_deps_b) == 1
        @test name(first(msg_deps_b)) === :in2
        
    end
end