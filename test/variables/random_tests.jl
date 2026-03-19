
@testitem "RandomVariable: uninitialized" begin
    import ReactiveMP: messageout, messagein

    # Should throw if not initialised properly
    let var = randomvar()
        for i in 1:10
            @test_throws BoundsError messageout(var, i)
            @test_throws BoundsError messagein(var, i)
        end
    end
end

@testitem "RandomVariable: getmessagein!" begin
    import ReactiveMP: MessageObservable, create_messagein!, messagein, degree

    # Test for different degrees `d`
    for d in 1:5:100
        let var = randomvar()
            for i in 1:d
                messagein, index = create_messagein!(var)
                @test messagein isa MessageObservable
                @test index === i
                @test degree(var) === i
            end
            @test degree(var) === d
        end
    end
end

@testitem "RandomVariable: getmarginal" begin
    import ReactiveMP:
        MessageObservable,
        MessageProductContext,
        create_messagein!,
        compute_product_of_messages,
        messagein,
        degree,
        activate!,
        connect!,
        RandomVariableActivationOptions,
        messageout

    include("../testutilities.jl")

    message_prod_fold = (variable, context, msgs) -> error("Messages should not be called here")
    marginal_prod_fold = (variable, context, msgs) -> msg(sum(getdata.(msgs)))
    for d in 1:5:100
        let var = randomvar()
            messageins = map(1:d) do _
                s = Subject(AbstractMessage)
                m, i = create_messagein!(var)
                connect!(m, s)
                return s
            end

            activate!(
                var,
                RandomVariableActivationOptions(
                    AsapScheduler(),
                    MessageProductContext(fold_strategy = message_prod_fold),
                    MessageProductContext(fold_strategy = marginal_prod_fold),
                ),
            )

            messages = map(msg, rand(d))

            marginal_expected = mgl(sum(getdata.(messages)))
            marginal_result = check_stream_updated_once(getmarginal(var)) do
                foreach(zip(messageins, messages)) do (messagein, message)
                    next!(messagein, message)
                end
            end

            # We check the `getdata` here approximatelly because the `marginal_prod_fn` can rearrange
            # the messages under the hood that introduces minor numerical differences
            @test getdata(marginal_result) ≈ getdata(marginal_expected)
        end
    end
end

@testitem "RandomVariable: messageout" begin
    import ReactiveMP:
        MessageObservable,
        MessageProductContext,
        create_messagein!,
        compute_product_of_messages,
        messagein,
        degree,
        activate!,
        connect!,
        RandomVariableActivationOptions,
        messageout

    include("../testutilities.jl")

    message_prod_fold = (variable, context, msgs) -> msg(sum(filter(!ismissing, getdata.(msgs))))
    marginal_prod_fold = (variable, context, msgs) -> error("Marginal should not be called here")

    # We start from `2` because `1` is not a valid degree for a random variable
    for d in 2:5:100, k in 1:d
        let var = randomvar()
            messageins = map(1:d) do _
                s = Subject(AbstractMessage)
                m, i = create_messagein!(var)
                connect!(m, s)
                return s
            end

            activate!(
                var,
                RandomVariableActivationOptions(
                    AsapScheduler(),
                    MessageProductContext(fold_strategy = message_prod_fold),
                    MessageProductContext(fold_strategy = marginal_prod_fold),
                ),
            )

            messages = map(msg, rand(d))

            # the outbound message is the result of multiplication of `n - 1` messages excluding index `k`
            kmessage_expected = msg(sum(filter(!ismissing, getdata.(collect(skipindex(messages, k))))))
            kmessage_result = check_stream_updated_once(messageout(var, k)) do
                foreach(zip(messageins, messages)) do (messagein, message)
                    next!(messagein, message)
                end
            end
            # We check the `getdata` here approximatelly because the `message_prod_fn` can rearrange
            # the messages under the hood that introduces minor numerical differences
            @test getdata(kmessage_result) ≈ getdata(kmessage_expected)
        end
    end
end

@testitem "RandomVariable: activate! - zero or less than one inbound messages should throw" begin
    import ReactiveMP: RandomVariableActivationOptions, activate!, messageout

    let var = randomvar()
        @test_throws "Cannot activate a random variable with zero or less than one inbound messages." activate!(
            var, RandomVariableActivationOptions()
        )
    end
end
