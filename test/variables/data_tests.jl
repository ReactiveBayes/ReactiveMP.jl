
@testitem "DataVariable: uninitialized" begin
    import ReactiveMP: messageout, messagein

    # Should throw if not initialised properly
    @testset let var = datavar()
        for i in 1:10
            @test messageout(var, 1) === messageout(var, i)
            @test_throws BoundsError messagein(var, i)
        end
    end
end

@testitem "DataVariable: getmessagein!" begin
    import ReactiveMP: MessageObservable, create_messagein!, messagein, degree

    # Test for different degrees `d`
    @testset for d in 1:5:100
        @testset let var = datavar()
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

@testitem "DataVariable: getmarginal" begin
    using BayesBase

    import ReactiveMP: MessageObservable, create_messagein!, messagein, degree, activate!, connect!, DataVariableActivationOptions, messageout

    include("../testutilities.jl")

    @testset begin
        # Test marginal computation
        @testset for d in 1:5:100
            @testset let var = datavar()
                messageins = map(1:d) do _
                    s = Subject(AbstractMessage)
                    m, i = create_messagein!(var)
                    connect!(m, s)
                    return s
                end

                activate!(var, DataVariableActivationOptions(false, false, nothing, nothing))

                messages = map(msg, rand(d))

                @test check_stream_not_updated(getmarginal(var)) do
                    foreach(zip(messageins, messages)) do (messagein, message)
                        next!(messagein, message)
                    end
                end

                data_point = rand()

                marginal_expected = mgl(PointMass(data_point))
                marginal_result = check_stream_updated_once(getmarginal(var)) do
                    update!(var, data_point)
                end

                @test getdata(marginal_result) === getdata(marginal_expected)
                @test getdata(marginal_result) === PointMass(data_point)
            end
        end
    end
end
