
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

@testitem "RandomVariable: activate!" begin
    import ReactiveMP: RandomVariable, MessageObservable, create_messagein!, messagein, degree, activate!, connect!, RandomVariableActivationOptions, messageout

    include("../testsutilities.jl")

    # Should throw if not initialised properly
    @testset let randomvar = RandomVariable()
        @test_throws BoundsError messageout(randomvar, 1)
        @test_throws BoundsError messagein(randomvar, 1)
    end

    @testset begin
        # Test marginal computation
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

                @test degree(randomvar) === d

                @test mgl(2.0 * d) == fetch_stream_updated(getmarginal(randomvar)) do
                    foreach(messageins) do messagein
                        next!(messagein, msg(2.0))
                    end
                end
            end
        end
    end

    @testset begin
        # Test messages computation
        message_prod_fn = (msgs) -> msg(sum(filter(!ismissing, getdata.(msgs))))
        marginal_prod_fn = (msgs) -> error("Marginal should not be called here")
        @testset for d in 2:5:100 # We start from `2` because `1` is not a valid degree for a random variable
            @testset let randomvar = RandomVariable(message_prod_fn, marginal_prod_fn)
                messageins = map(1:d) do _
                    s = Subject(AbstractMessage)
                    m, i = create_messagein!(randomvar)
                    connect!(m, s)
                    return s
                end
                activate!(randomvar, RandomVariableActivationOptions())

                @test degree(randomvar) === d

                foreach(1:d) do k
                    @test msg(2.0 * (d - 1)) == fetch_stream_updated(messageout(randomvar, k)) do
                        foreach(messageins) do messagein
                            next!(messagein, msg(2.0))
                        end
                    end
                end
            end
        end
    end
end
