
@testitem "RandomVariable: uninitialized" begin
    import ReactiveMP: RandomVariable, messageout, messagein

    # Should throw if not initialised properly
    @testset let randomvar = RandomVariable()
        for i in 1:10
            @test_throws BoundsError messageout(randomvar, i)
            @test_throws BoundsError messagein(randomvar, i)
        end
    end
end

@testitem "RandomVariable: getmessagein!" begin
    import ReactiveMP: RandomVariable, MessageObservable, create_messagein!, messagein, degree

    # Test for different degrees `d`
    @testset for d in 1:5:100
        @testset let randomvar = RandomVariable()
            for i in 1:d
                messagein, index = create_messagein!(randomvar)
                @test messagein isa MessageObservable
                @test index === i
                @test degree(randomvar) === i
            end
            @test degree(randomvar) === d
        end
    end
end

@testitem "RandomVariable: getmarginal" begin
    import ReactiveMP: RandomVariable, MessageObservable, create_messagein!, messagein, degree, activate!, connect!, RandomVariableActivationOptions, messageout

    include("../testutilities.jl")

    message_prod_fn = (msgs) -> error("Messages should not be called here")
    marginal_prod_fn = (msgs) -> mgl(sum(getdata.(msgs)))
    @testset for d in 1:5:100
        @testset let randomvar = RandomVariable(message_prod_fn, marginal_prod_fn)
            messageins = map(1:d) do _
                s = Subject(AbstractMessage)
                m, i = create_messagein!(randomvar)
                connect!(m, s)
                return s
            end

            activate!(randomvar, RandomVariableActivationOptions())

            messages = map(msg, rand(d))

            marginal_expected = marginal_prod_fn(messages)
            marginal_result = check_stream_updated_once(getmarginal(randomvar)) do
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
    import ReactiveMP: RandomVariable, MessageObservable, create_messagein!, messagein, degree, activate!, connect!, RandomVariableActivationOptions, messageout

    include("../testutilities.jl")

    message_prod_fn = (msgs) -> msg(sum(filter(!ismissing, getdata.(msgs))))
    marginal_prod_fn = (msgs) -> error("Marginal should not be called here")

    # We start from `2` because `1` is not a valid degree for a random variable
    @testset for d in 2:5:100, k in 1:d
        @testset let randomvar = RandomVariable(message_prod_fn, marginal_prod_fn)
            messageins = map(1:d) do _
                s = Subject(AbstractMessage)
                m, i = create_messagein!(randomvar)
                connect!(m, s)
                return s
            end

            activate!(randomvar, RandomVariableActivationOptions())

            messages = map(msg, rand(d))

            # the outbound message is the result of multiplication of `n - 1` messages excluding index `k`
            kmessage_expected = message_prod_fn(collect(skipindex(messages, k)))
            kmessage_result = check_stream_updated_once(messageout(randomvar, k)) do
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