
@testitem "DataVariable: uninitialized" begin
    import ReactiveMP: DataVariable, messageout, messagein

    # Should throw if not initialised properly
    @testset let datavar = DataVariable()
        for i in 1:10
            @test messageout(datavar, 1) === messageout(datavar, i)
            @test_throws BoundsError messagein(datavar, i)
        end
    end
end

@testitem "DataVariable: getmessagein!" begin
    import ReactiveMP: DataVariable, MessageObservable, create_messagein!, messagein, degree

    # Test for different degrees `d`
    @testset for d in 1:5:100
        @testset let datavar = DataVariable()
            for i in 1:d
                messagein, index = create_messagein!(datavar)
                @test messagein isa MessageObservable
                @test index === i
                @test degree(datavar) === i
            end
            @test degree(datavar) === d
        end
    end
end

@testitem "DataVariable: getmarginal" begin
    using BayesBase

    import ReactiveMP: DataVariable, MessageObservable, create_messagein!, messagein, degree, activate!, connect!, DataVariableActivationOptions, messageout

    include("../testutilities.jl")

    @testset begin
        # Test marginal computation
        @testset for d in 1:5:100
            @testset let datavar = DataVariable()
                messageins = map(1:d) do _
                    s = Subject(AbstractMessage)
                    m, i = create_messagein!(datavar)
                    connect!(m, s)
                    return s
                end

                activate!(datavar, DataVariableActivationOptions(false))

                messages = map(msg, rand(d))

                @test check_stream_not_updated(getmarginal(datavar)) do
                    foreach(zip(messageins, messages)) do (messagein, message)
                        next!(messagein, message)
                    end
                end

                data_point = rand()

                marginal_expected = mgl(PointMass(data_point))
                marginal_result = check_stream_updated_once(getmarginal(datavar)) do
                    update!(datavar, data_point)
                end

                @test getdata(marginal_result) === getdata(marginal_expected)
                @test getdata(marginal_result) === PointMass(data_point)
            end
        end
    end
end