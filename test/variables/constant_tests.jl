
@testitem "ConstVariable: uninitialized" begin
    import ReactiveMP:
        get_stream_of_outbound_messages, get_stream_of_inbound_messages

    # Should throw if not initialised properly
    let var = constvar(1)
        for i in 1:10
            @test get_stream_of_outbound_messages(var, 1) ===
                get_stream_of_outbound_messages(var, i)
            @test_throws ErrorException get_stream_of_inbound_messages(var, i)
        end
    end
end

@testitem "ConstVariable: get_stream_of_inbound_messages" begin
    import ReactiveMP:
        MessageObservable,
        create_new_stream_of_inbound_messages!,
        get_stream_of_inbound_messages,
        degree

    # Test for different degrees `d`
    for d in 1:5:100
        let var = constvar(1)
            for i in 1:d
                new_stream_of_inbound_messages, index = create_new_stream_of_inbound_messages!(
                    var
                )
                @test new_stream_of_inbound_messages isa MessageObservable
                @test index === 1
                @test degree(var) === i
            end
            @test degree(var) === d
        end
    end
end

@testitem "ConstVariable: get_stream_of_marginals" begin
    using BayesBase

    import ReactiveMP:
        MessageObservable,
        degree,
        activate!,
        connect!,
        DataVariableActivationOptions,
        get_stream_of_outbound_messages,
        get_stream_of_marginals

    include("../testutilities.jl")

    # Test marginal computation
    for d in 1:5:100, constant in rand(10)
        let var = constvar(constant)
            marginal_expected = mgl(PointMass(constant))
            marginal_result =
                check_stream_updated_once(get_stream_of_marginals(var)) do
                    nothing
                end

            @test getdata(marginal_result) === getdata(marginal_expected)
            @test getdata(marginal_result) === PointMass(constant)
        end
    end
end
