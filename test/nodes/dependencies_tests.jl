@testitem "collect_latest_messages" begin
    include("../testutilities.jl")
    using BayesBase

    import ReactiveMP: NodeInterface, collect_latest_messages, getdata, getrecent

    @testset let (tag, stream) = collect_latest_messages(())
        @test tag === nothing
        @test check_stream_updated_once(stream) === nothing
    end

    a = NodeInterface(:a, ConstVariable(1))
    b = NodeInterface(:b, ConstVariable(2))
    c = NodeInterface(:c, ConstVariable(3))

    @testset let (tag, stream) = collect_latest_messages((a, b, c))
        @test tag === Val((:a, :b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_messages((a, b))
        @test tag === Val((:a, :b))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2))
    end

    @testset let (tag, stream) = collect_latest_messages((b, c))
        @test tag === Val((:b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_messages((a, c))
        @test tag === Val((:a, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(3))
    end
end

@testitem "collect_latest_marginals" begin
    include("../testutilities.jl")
    using BayesBase

    import ReactiveMP: NodeInterface, FactorNodeLocalMarginal, getmarginal, collect_latest_marginals, getdata, getrecent

    struct ArbitraryNode end

    a = FactorNodeLocalMarginal(:a)
    b = FactorNodeLocalMarginal(:b)
    c = FactorNodeLocalMarginal(:c)

    setmarginal!(a, getmarginal(ConstVariable(1), IncludeAll()))
    setmarginal!(b, getmarginal(ConstVariable(2), IncludeAll()))
    setmarginal!(c, getmarginal(ConstVariable(3), IncludeAll()))

    @testset let (tag, stream) = collect_latest_marginals((a, b, c))
        @test tag === Val((:a, :b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_marginals((a, b))
        @test tag === Val((:a, :b))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2))
    end

    @testset let (tag, stream) = collect_latest_marginals((b, c))
        @test tag === Val((:b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_marginals((a, c))
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

        @testset "RequireMessageFunctionalDependencies(b = vague(NormalMeanPrecision))" begin
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

        @testset "RequireMessageFunctionalDependencies(b = vague(NormalMeanPrecision))" begin
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
end