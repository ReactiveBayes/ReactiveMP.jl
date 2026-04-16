
@testitem "RandomVariable: uninitialized" begin
    import ReactiveMP:
        get_stream_of_outbound_messages, get_stream_of_inbound_messages

    # Should throw if not initialised properly
    let var = randomvar()
        for i in 1:10
            @test_throws BoundsError get_stream_of_outbound_messages(var, i)
            @test_throws BoundsError get_stream_of_inbound_messages(var, i)
        end
    end
end

@testitem "RandomVariable: getget_stream_of_inbound_messages!" begin
    import ReactiveMP:
        MessageObservable,
        create_new_stream_of_inbound_messages!,
        get_stream_of_inbound_messages,
        degree

    # Test for different degrees `d`
    for d in 1:5:100
        let var = randomvar()
            for i in 1:d
                new_stream_of_inbound_messages, index = create_new_stream_of_inbound_messages!(
                    var
                )
                @test new_stream_of_inbound_messages isa MessageObservable
                @test index === i
                @test degree(var) === i
            end
            @test degree(var) === d
        end
    end
end

@testitem "RandomVariable: get_stream_of_marginals" begin
    import ReactiveMP:
        MessageObservable,
        MessageProductContext,
        NoopStreamPostprocessor,
        create_new_stream_of_inbound_messages!,
        compute_product_of_messages,
        get_stream_of_inbound_messages,
        degree,
        activate!,
        connect!,
        RandomVariableActivationOptions,
        get_stream_of_outbound_messages,
        get_stream_of_marginals

    include("../testutilities.jl")

    message_prod_fold =
        (variable, context, msgs) -> error("Messages should not be called here")
    marginal_prod_fold = (variable, context, msgs) -> msg(sum(getdata.(msgs)))
    for d in 1:5:100
        let var = randomvar()
            new_stream_of_inbound_messages = map(1:d) do _
                s = Subject(AbstractMessage)
                m, i = create_new_stream_of_inbound_messages!(var)
                connect!(m, s)
                return s
            end

            activate!(
                var,
                RandomVariableActivationOptions(
                    NoopStreamPostprocessor(),
                    MessageProductContext(; fold_strategy = message_prod_fold),
                    MessageProductContext(; fold_strategy = marginal_prod_fold),
                ),
            )

            messages = map(msg, rand(d))

            marginal_expected = mgl(sum(getdata.(messages)))
            marginal_result =
                check_stream_updated_once(get_stream_of_marginals(var)) do
                    foreach(
                        zip(new_stream_of_inbound_messages, messages)
                    ) do (new_stream_of_inbound_messages, message)
                        next!(new_stream_of_inbound_messages, message)
                    end
                end

            # We check the `getdata` here approximatelly because the `marginal_prod_fn` can rearrange
            # the messages under the hood that introduces minor numerical differences
            @test getdata(marginal_result) ≈ getdata(marginal_expected)
        end
    end
end

@testitem "RandomVariable: get_stream_of_outbound_messages" begin
    import ReactiveMP:
        MessageObservable,
        MessageProductContext,
        NoopStreamPostprocessor,
        create_new_stream_of_inbound_messages!,
        compute_product_of_messages,
        get_stream_of_inbound_messages,
        degree,
        activate!,
        connect!,
        RandomVariableActivationOptions,
        get_stream_of_outbound_messages

    include("../testutilities.jl")

    message_prod_fold =
        (variable, context, msgs) ->
            msg(sum(filter(!ismissing, getdata.(msgs))))
    marginal_prod_fold =
        (variable, context, msgs) -> error("Marginal should not be called here")

    # We start from `2` because `1` is not a valid degree for a random variable
    for d in 2:5:100, k in 1:d
        let var = randomvar()
            new_streams_of_inbound_messages = map(1:d) do _
                s = Subject(AbstractMessage)
                m, i = create_new_stream_of_inbound_messages!(var)
                connect!(m, s)
                return s
            end

            activate!(
                var,
                RandomVariableActivationOptions(
                    NoopStreamPostprocessor(),
                    MessageProductContext(; fold_strategy = message_prod_fold),
                    MessageProductContext(; fold_strategy = marginal_prod_fold),
                ),
            )

            messages = map(msg, rand(d))

            # the outbound message is the result of multiplication of `n - 1` messages excluding index `k`
            kmessage_expected = msg(
                sum(
                    filter(
                        !ismissing, getdata.(collect(skipindex(messages, k)))
                    ),
                ),
            )
            kmessage_result = check_stream_updated_once(
                get_stream_of_outbound_messages(var, k)
            ) do
                foreach(
                    zip(new_streams_of_inbound_messages, messages)
                ) do (new_stream_of_inbound_messages, message)
                    next!(new_stream_of_inbound_messages, message)
                end
            end
            # We check the `getdata` here approximatelly because the `message_prod_fn` can rearrange
            # the messages under the hood that introduces minor numerical differences
            @test getdata(kmessage_result) ≈ getdata(kmessage_expected)
        end
    end
end

@testitem "RandomVariable: before/after marginal computation callbacks" begin
    import ReactiveMP:
        MessageObservable,
        MessageProductContext,
        RandomVariableActivationOptions,
        NoopStreamPostprocessor,
        AbstractMessage,
        create_new_stream_of_inbound_messages!,
        activate!,
        connect!,
        getdata,
        get_stream_of_marginals

    import Rocket: Subject, next!

    include("../testutilities.jl")

    struct MarginalCallbackHandler
        listen_to::Tuple
        events
    end

    function ReactiveMP.invoke_callback(
        handler::MarginalCallbackHandler, event::ReactiveMP.Event{E}
    ) where {E}
        E ∈ handler.listen_to &&
            push!(handler.events, (event = E, data = event))
    end

    @testset "Fires before and after marginal computation with 3 messages" begin
        listen_to = (:before_marginal_computation, :after_marginal_computation)
        handler = MarginalCallbackHandler(listen_to, [])
        marginal_context = MessageProductContext(;
            fold_strategy = (variable, context, msgs) ->
                msg(sum(getdata.(msgs))),
            callbacks = handler,
        )

        var = randomvar()

        new_streams_of_inbounds_messages = map(1:3) do _
            s = Subject(AbstractMessage)
            m, i = create_new_stream_of_inbound_messages!(var)
            connect!(m, s)
            return s
        end

        activate!(
            var,
            RandomVariableActivationOptions(
                NoopStreamPostprocessor(),
                MessageProductContext(),
                marginal_context,
            ),
        )

        messages = [msg(1.0), msg(2.0), msg(3.0)]

        marginal_result =
            check_stream_updated_once(get_stream_of_marginals(var)) do
                foreach(
                    zip(new_streams_of_inbounds_messages, messages)
                ) do (new_stream_of_inbounds_messages, message)
                    next!(new_stream_of_inbounds_messages, message)
                end
            end

        # sum(1.0 + 2.0 + 3.0) = 6.0
        @test getdata(marginal_result) ≈ 6.0

        @test length(handler.events) == 2

        # Before: variable, context, messages
        @test handler.events[1].event === :before_marginal_computation
        @test handler.events[1].data.variable === var
        @test handler.events[1].data.context === marginal_context

        # After: variable, context, messages, result
        @test handler.events[2].event === :after_marginal_computation
        @test handler.events[2].data.variable === var
        @test handler.events[2].data.context === marginal_context
        @test length(handler.events[2].data.messages) == 3
        @test getdata(handler.events[2].data.result) ≈ 6.0
    end

    @testset "Fires before and after marginal computation with 2 messages" begin
        listen_to = (:before_marginal_computation, :after_marginal_computation)
        handler = MarginalCallbackHandler(listen_to, [])
        marginal_context = MessageProductContext(;
            fold_strategy = (variable, context, msgs) ->
                msg(sum(getdata.(msgs))),
            callbacks = handler,
        )

        var = randomvar()

        new_streams_of_inbounds_messages = map(1:2) do _
            s = Subject(AbstractMessage)
            m, i = create_new_stream_of_inbound_messages!(var)
            connect!(m, s)
            return s
        end

        activate!(
            var,
            RandomVariableActivationOptions(
                NoopStreamPostprocessor(),
                MessageProductContext(),
                marginal_context,
            ),
        )

        messages = [msg(10.0), msg(20.0)]

        marginal_result =
            check_stream_updated_once(get_stream_of_marginals(var)) do
                foreach(
                    zip(new_streams_of_inbounds_messages, messages)
                ) do (new_stream_of_inbound_messages, message)
                    next!(new_stream_of_inbound_messages, message)
                end
            end

        # sum(10.0 + 20.0) = 30.0
        @test getdata(marginal_result) ≈ 30.0

        @test length(handler.events) == 2

        @test handler.events[1].event === :before_marginal_computation
        @test handler.events[1].data.variable === var

        @test handler.events[2].event === :after_marginal_computation
        @test handler.events[2].data.variable === var
        @test length(handler.events[2].data.messages) == 2
        @test getdata(handler.events[2].data.result) ≈ 30.0
    end
end

@testitem "RandomVariable: activate! - zero or less than one inbound messages should throw" begin
    import ReactiveMP:
        RandomVariableActivationOptions,
        activate!,
        get_stream_of_outbound_messages

    let var = randomvar()
        @test_throws "Cannot activate a random variable with zero or less than one inbound messages." activate!(
            var, RandomVariableActivationOptions()
        )
    end
end
