
@testitem "DataVariable: uninitialized" begin
    import ReactiveMP:
        get_stream_of_outbound_messages, get_stream_of_inbound_messages

    # Should throw if not initialised properly
    let var = datavar()
        for i in 1:10
            @test get_stream_of_outbound_messages(var, 1) ===
                get_stream_of_outbound_messages(var, i)
            @test_throws BoundsError get_stream_of_inbound_messages(var, i)
        end
    end
end

@testitem "DataVariable: get_stream_of_inbound_messages" begin
    import ReactiveMP:
        MessageObservable,
        create_new_stream_of_inbound_messages!,
        get_stream_of_inbound_messages,
        degree

    # Test for different degrees `d`
    for d in 1:5:100
        let var = datavar()
            for i in 1:d
                new_stream_of_inbound_message, index = create_new_stream_of_inbound_messages!(
                    var
                )
                @test new_stream_of_inbound_message isa MessageObservable
                @test index === i
                @test degree(var) === i
            end
            @test degree(var) === d
        end
    end
end

@testitem "DataVariable: getmarginal" begin
    using BayesBase

    import ReactiveMP:
        MessageObservable,
        create_new_stream_of_inbound_messages!,
        get_stream_of_inbound_messages,
        degree,
        activate!,
        connect!,
        new_observation!,
        DataVariableActivationOptions,
        get_stream_of_outbound_messages,
        get_stream_of_marginals

    include("../testutilities.jl")

    for d in 1:5:100
        let var = datavar()
            new_streams_of_inbound_messages = map(1:d) do _
                s = Subject(AbstractMessage)
                m, i = create_new_stream_of_inbound_messages!(var)
                connect!(m, s)
                return s
            end

            activate!(
                var,
                DataVariableActivationOptions(false, false, nothing, nothing),
            )

            messages = map(msg, rand(d))

            @test check_stream_not_updated(get_stream_of_marginals(var)) do
                foreach(
                    zip(new_streams_of_inbound_messages, messages)
                ) do (new_stream_of_inbound_messages, message)
                    next!(new_stream_of_inbound_messages, message)
                end
            end

            data_point = rand()

            marginal_expected = mgl(PointMass(data_point))
            marginal_result = check_stream_updated_once(
                get_stream_of_marginals(var)
            ) do
                new_observation!(var, data_point)
            end

            @test getdata(marginal_result) === getdata(marginal_expected)
            @test getdata(marginal_result) === PointMass(data_point)
        end
    end
end

@testitem "DataVariable: linking to a non-PointMass marginal gives an informative error" begin
    using BayesBase, Distributions, ExponentialFamily
    import ReactiveMP:
        DataVariable,
        DataVariableActivationOptions,
        RandomVariable,
        RandomVariableActivationOptions,
        activate!,
        get_stream_of_marginals,
        __apply_link,
        __apply_link_data

    include("../testutilities.jl")

    # A linked data variable applies its transform to *point* values. Linking it to
    # something whose marginal is a distribution used to fail with a bare `MethodError`
    # naming only the internal `__apply_link`, which tells the user nothing about what
    # they did wrong (issue #634).

    @testset "the error names the offending argument and its type" begin
        err = try
            __apply_link_data(+, (PointMass(1.0), NormalMeanVariance(0.0, 1.0)))
            nothing
        catch e
            e
        end

        @test err isa ErrorException
        @test occursin("must resolve to a `PointMass`", err.msg)
        # Points at the specific argument, by position and by type.
        @test occursin("argument 2", err.msg)
        @test occursin("NormalMeanVariance", err.msg)
        # Does not mention the first argument, which was fine.
        @test !occursin("argument 1", err.msg)
        # Explains why, and what to do instead.
        @test occursin("random variable", err.msg)
        @test occursin("new_observation!", err.msg)
    end

    @testset "every offending argument is listed" begin
        err = try
            __apply_link_data(
                +,
                (
                    NormalMeanVariance(0.0, 1.0),
                    PointMass(1.0),
                    Gamma(1.0, 1.0),
                ),
            )
            nothing
        catch e
            e
        end

        @test err isa ErrorException
        @test occursin("argument 1", err.msg)
        @test occursin("argument 3", err.msg)
        @test !occursin("argument 2", err.msg)
    end

    @testset "reached through the real entry point, which receives Marginals" begin
        # `activate!` wires `__apply_link(f, getrecent.(args))`, where each element is the
        # `Marginal` most recently emitted by a linked variable's marginal stream. This is the
        # path an actual model takes, so drive it with `Marginal`s rather than raw data.
        err = try
            __apply_link(
                +,
                (
                    Marginal(PointMass(1.0), true, false),
                    Marginal(NormalMeanVariance(0.0, 1.0), false, false),
                ),
            )
            nothing
        catch e
            e
        end

        @test err isa ErrorException
        @test occursin("must resolve to a `PointMass`", err.msg)
        @test occursin("argument 2", err.msg)
    end

    @testset "the all-PointMass path is unaffected" begin
        # The valid case must still work, and must not go anywhere near the error branch.
        @test __apply_link_data(+, (PointMass(2.0), PointMass(3.0))) == 5.0
        @test __apply_link(
            *,
            (
                Marginal(PointMass(2.0), true, false),
                Marginal(PointMass(3.0), true, false),
            ),
        ) == 6.0
    end
end

@testitem "DataVariable: linked variable" begin
    using BayesBase
    import ReactiveMP:
        DataVariable,
        DataVariableActivationOptions,
        activate!,
        get_stream_of_outbound_messages,
        get_stream_of_marginals,
        new_observation!

    include("../testutilities.jl")

    for fn in (+, *), val1 in 1:3, val2 in 1:3
        let var = datavar()
            options = DataVariableActivationOptions(
                true, true, fn, (val1, val2)
            )
            activate!(var, options)
            marginal = check_stream_updated_once(get_stream_of_marginals(var))
            @test getdata(marginal) === PointMass(fn(val1, val2))
            message = check_stream_updated_once(
                get_stream_of_outbound_messages(var, 1)
            )
            @test getdata(message) === PointMass(fn(val1, val2))
        end

        # Just marginal
        let var = datavar()
            options = DataVariableActivationOptions(
                true, true, fn, (val1, val2)
            )
            activate!(var, options)
            marginal = check_stream_updated_once(get_stream_of_marginals(var))
            @test getdata(marginal) === PointMass(fn(val1, val2))
        end

        # Just message
        let var = datavar()
            options = DataVariableActivationOptions(
                true, true, fn, (val1, val2)
            )
            activate!(var, options)
            message = check_stream_updated_once(
                get_stream_of_outbound_messages(var, 1)
            )
            @test getdata(message) === PointMass(fn(val1, val2))
        end

        let
            var1 = datavar()
            activate!(
                var1,
                DataVariableActivationOptions(true, false, nothing, nothing),
            )

            var = datavar()
            options = DataVariableActivationOptions(
                true, true, fn, (var1, val2)
            )
            activate!(var, options)
            @test check_stream_not_updated(get_stream_of_marginals(var))

            marginal = check_stream_updated_once(
                get_stream_of_marginals(var)
            ) do
                new_observation!(var1, val1)
            end
            @test getdata(marginal) === PointMass(fn(val1, val2))
            message = check_stream_updated_once(
                get_stream_of_outbound_messages(var, 1)
            )
            @test getdata(message) === PointMass(fn(val1, val2))
        end

        let
            var2 = datavar()
            activate!(
                var2,
                DataVariableActivationOptions(true, false, nothing, nothing),
            )

            var = datavar()
            options = DataVariableActivationOptions(
                true, true, fn, (val1, var2)
            )
            activate!(var, options)
            @test check_stream_not_updated(get_stream_of_marginals(var))

            marginal = check_stream_updated_once(
                get_stream_of_marginals(var)
            ) do
                new_observation!(var2, val2)
            end
            @test getdata(marginal) === PointMass(fn(val1, val2))

            message = check_stream_updated_once(
                get_stream_of_outbound_messages(var, 1)
            )
            @test getdata(message) === PointMass(fn(val1, val2))
        end

        let
            var1 = datavar()
            var2 = datavar()
            activate!(
                var1,
                DataVariableActivationOptions(true, false, nothing, nothing),
            )
            activate!(
                var2,
                DataVariableActivationOptions(true, false, nothing, nothing),
            )

            var = datavar()
            options = DataVariableActivationOptions(
                true, true, fn, (var1, var2)
            )
            activate!(var, options)
            @test check_stream_not_updated(get_stream_of_marginals(var))

            marginal = check_stream_updated_once(
                get_stream_of_marginals(var)
            ) do
                new_observation!(var1, val1)
                new_observation!(var2, val2)
            end
            @test getdata(marginal) === PointMass(fn(val1, val2))

            message = check_stream_updated_once(
                get_stream_of_outbound_messages(var, 1)
            )
            @test getdata(message) === PointMass(fn(val1, val2))
        end

        let
            var1 = datavar()
            var2 = datavar()
            activate!(
                var1,
                DataVariableActivationOptions(true, false, nothing, nothing),
            )
            activate!(
                var2,
                DataVariableActivationOptions(true, false, nothing, nothing),
            )

            var = datavar()
            options = DataVariableActivationOptions(
                true, true, fn, (var1, var2)
            )
            activate!(var, options)
            @test check_stream_not_updated(get_stream_of_marginals(var))

            # We still should be able to update the stream manually
            marginal = check_stream_updated_once(
                get_stream_of_marginals(var)
            ) do
                new_observation!(var, 4)
            end
            @test getdata(marginal) === PointMass(4)
        end
    end
end
