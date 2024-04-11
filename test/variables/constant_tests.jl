
@testitem "ConstVariable: uninitialized" begin
    import ReactiveMP: messageout, messagein

    # Should throw if not initialised properly
    @testset let var = constvar(1)
        for i in 1:10
            @test messageout(var, 1) === messageout(var, i)
            @test_throws ErrorException messagein(var, i)
        end
    end
end

@testitem "ConstVariable: getmessagein!" begin
    import ReactiveMP: MessageObservable, create_messagein!, messagein, degree

    # Test for different degrees `d`
    @testset for d in 1:5:100
        @testset let var = constvar(1)
            for i in 1:d
                messagein, index = create_messagein!(var)
                @test messagein isa MessageObservable
                @test index === 1
                @test degree(var) === i
            end
            @test degree(var) === d
        end
    end
end

@testitem "ConstVariable: getmarginal" begin
    using BayesBase

    import ReactiveMP: MessageObservable, create_messagein!, messagein, degree, activate!, connect!, DataVariableActivationOptions, messageout

    include("../testutilities.jl")

    @testset begin
        # Test marginal computation
        @testset for d in 1:5:100, constant in rand(10)
            @testset let var = constvar(constant)
                marginal_expected = mgl(PointMass(constant))
                marginal_result = check_stream_updated_once(getmarginal(var)) do
                    nothing
                end

                @test getdata(marginal_result) === getdata(marginal_expected)
                @test getdata(marginal_result) === PointMass(constant)
            end
        end
    end
end
