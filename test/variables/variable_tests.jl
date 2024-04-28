
@testitem "Variable" begin
    using ReactiveMP, Rocket, BayesBase, Distributions, ExponentialFamily

    import ReactiveMP: activate!
    import Rocket: getscheduler

    struct CustomDeterministicNode end

    @node CustomDeterministicNode Deterministic [out, x]

    function test_variable_set_method(variable, dist::T, k) where {T}
        test_out_var = randomvar()

        # messages could be initialized only when the node is created
        for _ in 1:k
            factornode(CustomDeterministicNode, [(:out, test_out_var), (:x, variable)], ((1, 2),))
        end

        activate!(test_out_var, RandomVariableActivationOptions())
        activate!(variable, RandomVariableActivationOptions())

        @test degree(variable) === k

        # Check that before calling the `setmarginals!` all marginals are `nothing`
        @test isnothing(Rocket.getrecent(getmarginal(variable, IncludeAll())))

        setmarginal!(variable, dist)

        marginal_subscription_flag = false
        # After calling the `setmarginals!` the marginal should be equal to `dist`
        subscription = subscribe!(getmarginal(variable, IncludeAll()), (marginal) -> begin
            @test typeof(marginal) <: Marginal{T}
            @test mean(marginal) === mean(dist)
            @test var(marginal) === var(dist)
            marginal_subscription_flag = true
        end)
        @test marginal_subscription_flag === true
        unsubscribe!(subscription)

        # Check that before calling the `setmessages!` all messages are `nothing`
        for node_index in 1:k
            @test isnothing(Rocket.getrecent(ReactiveMP.messageout(variable, node_index)))
        end

        for node_index in 1:k
            setmessage!(variable, node_index, dist)
        end

        for node_index in 1:k
            message_subscription_flag = false
            subscription = subscribe!(ReactiveMP.messageout(variable, node_index), (message) -> begin
                @test typeof(message) <: Message{T}
                @test mean(message) === mean(dist)
                @test var(message) === var(dist)
                message_subscription_flag = true
            end)
            @test message_subscription_flag === true
            unsubscribe!(subscription)
        end
    end

    function test_variables_set_methods(variables, dist::T, k::Int) where {T}
        marginal_subscription_flag = false

        @test_throws AssertionError setmarginals!(variables, Iterators.repeated(dist, length(variables) - 1))

        test_out_var = randomvar()

        for _ in 1:k
            factornode((x) -> sum(x...), [(:out, test_out_var), map(var -> (:in, var), variables)...], nothing)
        end

        foreach(var -> activate!(var, RandomVariableActivationOptions()), variables)

        @test all(degree.(variables) .== k)

        @test_throws AssertionError setmessages!(variables, Iterators.repeated(dist, length(variables) - 1))
        @test_throws AssertionError setmessages!(variables, Iterators.repeated(dist, length(variables) - 1))

        # Test `setmarginals!`

        # Check that before calling the `setmarginals!` all marginals are `nothing`
        @test all(isnothing, Rocket.getrecent.(getmarginal.(variables, IncludeAll())))

        setmarginals!(variables, dist)

        # After calling the `setmarginals!` all marginals should be equal to `dist`
        subscription = subscribe!(getmarginals(variables, IncludeAll()), (marginals) -> begin
            @test length(marginals) === length(variables)
            foreach(marginals) do marginal
                @test typeof(marginal) <: Marginal{T}
                @test mean(marginal) === mean(dist)
                @test var(marginal) === var(dist)
            end
            marginal_subscription_flag = true
        end)

        # Test that subscription happenend
        @test marginal_subscription_flag === true
        unsubscribe!(subscription)

        # Check that before calling the `setmessages!` all messages are `nothing`
        for node_index in 1:k
            @test all(isnothing, Rocket.getrecent.(ReactiveMP.messageout.(variables, node_index)))
        end

        # After calling the `setmessages!` all marginals should be equal to `dist`
        setmessages!(variables, dist)
        # For each outbound index
        for node_index in 1:k
            messages_subscription_flag = false
            subscription = subscribe!(collectLatest(ReactiveMP.messageout.(variables, node_index)), (messages) -> begin
                @test length(messages) === length(variables)
                foreach(messages) do message
                    @test typeof(message) <: Message{T}
                    @test mean(message) === mean(dist)
                    @test var(message) === var(dist)
                end
                messages_subscription_flag = true
            end)
            @test messages_subscription_flag === true
            unsubscribe!(subscription)
        end
    end

    @testset "setmarginal! and setmessages! tests for randomvar" begin
        dists = (NormalMeanVariance(-2.0, 3.0), NormalMeanPrecision(-2.0, 3.0), PointMass(2.0))
        number_of_nodes = 1:4
        for (dist, k) in Iterators.product(dists, number_of_nodes)
            test_variable_set_method(randomvar(), dist, k)
            test_variables_set_methods(map(_ -> randomvar(), ones(2)), dist, k)
            test_variables_set_methods(map(_ -> randomvar(), ones(2, 2)), dist, k)
        end
    end
end
