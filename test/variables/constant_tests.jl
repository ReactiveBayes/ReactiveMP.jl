
@testitem "ConstVariable: uninitialized" begin
    import ReactiveMP: ConstVariable, messageout, messagein

    # Should throw if not initialised properly
    @testset let constvar = ConstVariable(1)
        for i in 1:10
            @test messageout(constvar, 1) === messageout(constvar, i)
            @test_throws ErrorException messagein(constvar, i)
        end
    end
end

@testitem "ConstVariable: getmessagein!" begin
    import ReactiveMP: ConstVariable, MessageObservable, create_messagein!, messagein, degree

    # Test for different degrees `d`
    @testset for d in 1:5:100
        @testset let constvar = ConstVariable(1)
            for i in 1:d
                messagein, index = create_messagein!(constvar)
                @test messagein isa MessageObservable
                @test index === 1
                @test degree(constvar) === i
            end
            @test degree(constvar) === d
        end
    end
end

@testitem "ConstVariable: getmarginal" begin
    using BayesBase

    import ReactiveMP: ConstVariable, MessageObservable, create_messagein!, messagein, degree, activate!, connect!, DataVariableActivationOptions, messageout

    include("../testutilities.jl")

    @testset begin
        # Test marginal computation
        @testset for d in 1:5:100, constant in rand(10)
            @testset let constvar = ConstVariable(constant)

                marginal_expected = mgl(PointMass(constant))
                marginal_result = check_stream_updated_once(getmarginal(constvar)) do
                    nothing
                end

                @test getdata(marginal_result) === getdata(marginal_expected)
                @test getdata(marginal_result) === PointMass(constant)
            end
        end
    end
end