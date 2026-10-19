@testitem "RandomVariable: uninitialized" tags = [:engine] begin
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

@testitem "RandomVariable: getget_stream_of_inbound_messages!" tags = [:engine] begin
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

@testitem "RandomVariable: get_stream_of_marginals" tags = [:engine] begin
    import ReactiveMP:
        MessageObservable,
        MessageProductContext,
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
                    nothing,
                    MessageProductContext(; fold_strategy = message_prod_fold),
                    MessageProductContext(; fold_strategy = marginal_prod_fold),
                ),
            )

            messages = map(msg, rand(d))

            marginal_expected = mgl(sum(getdata.(messages)))
            marginal_result = check_stream_updated_once(
                get_stream_of_marginals(var)
            ) do
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

@testitem "RandomVariable: get_stream_of_outbound_messages" tags = [:engine] begin
    import ReactiveMP:
        MessageObservable,
        MessageProductContext,
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
                    nothing,
                    MessageProductContext(; fold_strategy = message_prod_fold),
                    MessageProductContext(; fold_strategy = marginal_prod_fold),
                ),
            )

            messages = map(msg, rand(d))

            # the outbound message is the result of multiplication of `n - 1` messages excluding index `k`
            kmessage_expected = msg(
                sum(
                    filter(
                        !ismissing, getdata.(messages[setdiff(eachindex(messages), k)])
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

@testitem "RandomVariable: before/after marginal computation callbacks" tags = [
    :engine,
] begin
    import ReactiveMP:
        MessageObservable,
        MessageProductContext,
        RandomVariableActivationOptions,
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
                nothing, MessageProductContext(), marginal_context
            ),
        )

        messages = [msg(1.0), msg(2.0), msg(3.0)]

        marginal_result = check_stream_updated_once(
            get_stream_of_marginals(var)
        ) do
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
                nothing, MessageProductContext(), marginal_context
            ),
        )

        messages = [msg(10.0), msg(20.0)]

        marginal_result = check_stream_updated_once(
            get_stream_of_marginals(var)
        ) do
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

@testitem "RandomVariable: activate! - zero or less than one inbound messages should throw" tags = [
    :engine,
] begin
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

@testitem "RandomVariable: a message form constraint applies once per outbound message under FormConstraintCheckLast" tags = [
    :engine,
] begin
    import ReactiveMP:
        MessageProductContext,
        RandomVariableActivationOptions,
        AbstractFormConstraint,
        FormConstraintCheckLast,
        FormConstraintCheckEach,
        create_new_stream_of_inbound_messages!,
        get_stream_of_outbound_messages,
        activate!,
        connect!,
        getdata

    using BayesBase, Distributions, ExponentialFamily
    import Rocket: Subject, next!, subscribe!, unsubscribe!

    # Doubles the variance: applying it twice differs from applying it once, so the result says
    # how often it was applied.
    struct DoubleVariance <: AbstractFormConstraint end
    ReactiveMP.constrain_form(::DoubleVariance, d) = NormalMeanVariance(mean(d), 2 * var(d))

    struct ProductEvents
        counts::Dict{Symbol, Int}
    end
    ReactiveMP.invoke_callback(handler::ProductEvents, ::ReactiveMP.Event{E}) where {E} =
        (handler.counts[E] = get(handler.counts, E, 0) + 1; nothing)

    product(ds...) = let w = sum(d -> 1 / var(d), ds)
        NormalMeanVariance(sum(d -> mean(d) / var(d), ds) / w, 1 / w)
    end
    f(d) = NormalMeanVariance(mean(d), 2 * var(d))

    μ = [NormalMeanVariance(1.0, 1.0), NormalMeanVariance(2.0, 2.0), NormalMeanVariance(-1.0, 0.5), NormalMeanVariance(0.5, 4.0)]

    function outbound_messages(strategy)
        handler = ProductEvents(Dict{Symbol, Int}())
        context = MessageProductContext(; form_constraint = DoubleVariance(), form_constraint_check_strategy = strategy, callbacks = handler)
        var = randomvar()
        inbound = map(1:4) do _
            s = Subject(AbstractMessage)
            m, _ = create_new_stream_of_inbound_messages!(var)
            connect!(m, s)
            return s
        end
        activate!(var, RandomVariableActivationOptions(nothing, context, MessageProductContext()))
        latest = Vector{Any}(missing, 4)
        emissions = Ref(0)
        subscriptions = map(1:4) do k
            subscribe!(get_stream_of_outbound_messages(var, k), (m) -> (latest[k] = getdata(m); emissions[] += 1))
        end
        foreach(((s, d),) -> next!(s, Message(d, false, false)), zip(inbound, μ))
        foreach(unsubscribe!, subscriptions)
        return latest, emissions[], handler.counts
    end

    # Under CheckLast the message to connection k is f of the product of the other three.
    latest, emissions, counts = outbound_messages(FormConstraintCheckLast())
    for k in 1:4
        expected = f(product(μ[setdiff(1:4, k)]...))
        @test mean(latest[k]) ≈ mean(expected) && var(latest[k]) ≈ var(expected)
    end
    # One whole product, and one application of the constraint, per outbound message computed.
    @test counts[:before_product_of_messages] == emissions
    @test counts[:after_product_of_messages] == emissions
    @test counts[:before_form_constraint_applied] == emissions

    # Under CheckEach every pairwise product is constrained, the partial products the chain
    # caches included: the message to connection 2 is f(f(μ₁) f(μ₃ f(μ₄))).
    latest, _, _ = outbound_messages(FormConstraintCheckEach())
    expected = f(product(f(μ[1]), f(product(μ[3], f(μ[4])))))
    @test mean(latest[2]) ≈ mean(expected) && var(latest[2]) ≈ var(expected)
end
