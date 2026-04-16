
@testitem "Variable" begin
    using ReactiveMP, Rocket, BayesBase, Distributions, ExponentialFamily

    import ReactiveMP:
        activate!,
        get_stream_of_marginals,
        degree,
        set_initial_marginal!,
        set_initial_message!

    struct CustomDeterministicNodeForVariableTests end

    @node CustomDeterministicNodeForVariableTests Deterministic [out, x]

    function test_variable_set_method(variable, dist::T, k) where {T}
        test_out_var = randomvar()

        # messages could be initialized only when the node is created
        for _ in 1:k
            factornode(
                CustomDeterministicNodeForVariableTests,
                [(:out, test_out_var), (:x, variable)],
                ((1, 2),),
            )
        end

        activate!(test_out_var, RandomVariableActivationOptions())
        activate!(variable, RandomVariableActivationOptions())

        @test degree(variable) === k

        # Check that before calling the `set_initial_marginal!` all marginals are `nothing`
        @test isnothing(Rocket.getrecent(get_stream_of_marginals(variable)))

        set_initial_marginal!(variable, dist)

        marginal_subscription_flag = false
        # After calling the `set_initial_marginal!` the marginal should be equal to `dist`
        subscription = subscribe!(
            get_stream_of_marginals(variable),
            (marginal) -> begin
                @test typeof(marginal) <: Marginal{T}
                @test mean(marginal) === mean(dist)
                @test var(marginal) === var(dist)
                marginal_subscription_flag = true
            end,
        )
        @test marginal_subscription_flag === true
        unsubscribe!(subscription)

        # Check that before calling the `set_initial_message!` all messages are `nothing`
        for node_index in 1:k
            @test isnothing(
                Rocket.getrecent(
                    ReactiveMP.get_stream_of_outbound_messages(
                        variable, node_index
                    ),
                ),
            )
        end

        set_initial_message!(variable, dist)

        for node_index in 1:k
            message_subscription_flag = false
            subscription = subscribe!(
                ReactiveMP.get_stream_of_outbound_messages(
                    variable, node_index
                ),
                (message) -> begin
                    @test typeof(message) <: Message{T}
                    @test mean(message) === mean(dist)
                    @test var(message) === var(dist)
                    message_subscription_flag = true
                end,
            )
            @test message_subscription_flag === true
            unsubscribe!(subscription)
        end
    end

    function test_variables_set_methods(variables, dist::T, k::Int) where {T}
        marginal_subscription_flag = false

        @test_throws AssertionError set_initial_marginal!(
            variables, Iterators.repeated(dist, length(variables) - 1)
        )

        test_out_var = randomvar()

        for _ in 1:k
            factornode(
                (x) -> sum(x...),
                [(:out, test_out_var), map(var -> (:in, var), variables)...],
                nothing,
            )
        end

        foreach(
            var -> activate!(var, RandomVariableActivationOptions()), variables
        )

        @test all(degree.(variables) .== k)

        @test_throws AssertionError set_initial_message!(
            variables, Iterators.repeated(dist, length(variables) - 1)
        )
        @test_throws AssertionError set_initial_message!(
            variables, Iterators.repeated(dist, length(variables) - 1)
        )

        # Test `set_initial_marginal!`

        # Check that before calling the `set_initial_marginal!` all marginals are `nothing`
        @test all(
            isnothing, Rocket.getrecent.(get_stream_of_marginals.(variables))
        )

        set_initial_marginal!(variables, dist)

        # After calling the `set_initial_marginal!` all marginals should be equal to `dist`
        subscription = subscribe!(
            collectLatest(map(get_stream_of_marginals, variables)),
            (marginals) -> begin
                @test length(marginals) === length(variables)
                foreach(marginals) do marginal
                    @test typeof(marginal) <: Marginal{T}
                    @test mean(marginal) === mean(dist)
                    @test var(marginal) === var(dist)
                end
                marginal_subscription_flag = true
            end,
        )

        # Test that subscription happenend
        @test marginal_subscription_flag === true
        unsubscribe!(subscription)

        # Check that before calling the `set_initial_message!` all messages are `nothing`
        for node_index in 1:k
            @test all(
                isnothing,
                Rocket.getrecent.(
                    ReactiveMP.get_stream_of_outbound_messages.(
                        variables, node_index
                    ),
                ),
            )
        end

        # After calling the `set_initial_message!` all marginals should be equal to `dist`
        set_initial_message!(variables, dist)
        # For each outbound index
        for node_index in 1:k
            messages_subscription_flag = false
            subscription = subscribe!(
                collectLatest(
                    ReactiveMP.get_stream_of_outbound_messages.(
                        variables, node_index
                    ),
                ),
                (messages) -> begin
                    @test length(messages) === length(variables)
                    foreach(messages) do message
                        @test typeof(message) <: Message{T}
                        @test mean(message) === mean(dist)
                        @test var(message) === var(dist)
                    end
                    messages_subscription_flag = true
                end,
            )
            @test messages_subscription_flag === true
            unsubscribe!(subscription)
        end
    end

    @testset "set_initial_marginal! and set_initial_message! tests for randomvar" begin
        dists = (
            NormalMeanVariance(-2.0, 3.0),
            NormalMeanPrecision(-2.0, 3.0),
            PointMass(2.0),
        )
        number_of_nodes = 1:4
        for (dist, k) in Iterators.product(dists, number_of_nodes)
            test_variable_set_method(randomvar(), dist, k)
            test_variables_set_methods(map(_ -> randomvar(), ones(2)), dist, k)
            test_variables_set_methods(
                map(_ -> randomvar(), ones(2, 2)), dist, k
            )
        end
    end
end
