
@testitem "DeltaNode - creation with static inputs (simple case) #1" begin
    using Rocket
    import ReactiveMP: nodefunction, DeltaMeta, Linearization, messageout, activate!, RandomVariableActivationOptions, DataVariableActivationOptions

    foo(x, y, z) = x * y + z

    out = randomvar()

    x = randomvar()
    y = datavar()
    z = constvar(3.0)

    node = factornode(foo, [(:out, out), (:in, x), (:in, y), (:in, z)], ((1, 2, 3, 4),))
    meta = DeltaMeta(method = Linearization())

    activate!(x, RandomVariableActivationOptions())
    activate!(y, DataVariableActivationOptions())

    update!(y, 2.0)

    for xval in rand(10)
        @test nodefunction(node, meta, Val(:out))(xval) === foo(xval, 2.0, 3.0)
        @test nodefunction(node)(foo(xval, 2.0, 3.0), xval) === 1
        @test nodefunction(node)(foo(xval, 2.0, 3.0) + 1.0, xval) === 0
    end
end

@testitem "DeltaNode - Creation with static inputs (all permutations) #2" begin
    using Rocket
    import ReactiveMP: nodefunction, DeltaMeta, Linearization, messageout, activate!, RandomVariableActivationOptions, DataVariableActivationOptions

    foo1(x, y, z) = x * y + z
    foo2(x, y, z) = x / y - z
    foo3(x, y, z) = x - y * z

    for vals in [rand(Float64, 3) for _ in 1:10], foo in (foo1, foo2, foo3)

        # In this test we attempt to create a lot of possible combinations 
        # of random, data and const inputs to the delta node, we make deepcopies 
        # of random variables to avoid reusing them between different tests
        function create_interfaces(i)
            r = randomvar()
            d = datavar()
            c = constvar(vals[i])
            return ((:in, r), (:in, d), (:in, c))
        end

        function activate_interface(r::RandomVariable)
            activate!(r, RandomVariableActivationOptions())
        end

        function activate_interface(d::DataVariable)
            activate!(d, DataVariableActivationOptions())
        end

        function activate_interface(c::ConstVariable)
            nothing
        end

        for x in create_interfaces(1), y in create_interfaces(2), z in create_interfaces(3)
            out_interface = (:out, randomvar())
            in_interfaces = [deepcopy(x), deepcopy(y), deepcopy(z)]
            interfaces = [out_interface, in_interfaces...]

            rpos = findall(i -> i isa Tuple{Symbol, RandomVariable}, in_interfaces)
            node = factornode(foo, interfaces, ((1, 2, 3, 4),))
            meta = DeltaMeta(method = Linearization())

            foreach(interfaces) do (_, interface)
                activate_interface(interface)
            end

            # data variable inputs require an actual update
            foreach(enumerate(in_interfaces)) do (i, interface)
                if interface isa Tuple{Symbol, DataVariable}
                    update!(interface[2], vals[i])
                end
            end

            @test nodefunction(node, meta, Val(:out))(vals[rpos]...) === foo(vals...)
            @test nodefunction(node)(foo(vals...), vals[rpos]...) === 1
            @test nodefunction(node)(foo(vals...) + 1, vals[rpos]...) === 0
        end
    end
end

@testitem "Unssupported methods should throw in DeltaMeta" begin
    struct UnsupportedApproximationMethod end

    @test_throws "Method `$(UnsupportedApproximationMethod())` is not compatible with delta nodes" DeltaMeta(method = UnsupportedApproximationMethod())
end

@testitem "Supported methods should not throw in DeltaMeta" begin
    struct SupportedApproximationMetßhod end

    ReactiveMP.is_delta_node_compatible(::SupportedApproximationMetßhod) = Val(true)

    @test DeltaMeta(method = SupportedApproximationMetßhod()) isa DeltaMeta
end

@testitem "DeltaNode - CVI layout functionality" begin
    using Rocket
    import BayesBase
    import ReactiveMP: DeltaFn, DeltaFnNode, DeltaMeta, CVIProjection, messageout, activate!, RandomVariableActivationOptions, DataVariableActivationOptions

    # Define a simple function for the delta node
    f(x, y) = x + y

    # Create variables
    out = randomvar()
    x   = randomvar()
    y   = datavar()

    # Create node with two random and one data input
    node = factornode(f, [(:out, out), (:in, x), (:in, y)], ((1, 2, 3),))

    # Test meta creation and compatibility
    meta = DeltaMeta(method = CVIProjection())
    @test meta.method isa CVIProjection
    @test isnothing(meta.inverse)

    # Activate variables with default options
    activate!(x, RandomVariableActivationOptions())
    activate!(y, DataVariableActivationOptions())

    # Test data variable update propagation
    update!(y, 2.0)
    @test BayesBase.getpointmass(getdata(Rocket.getrecent(messageout(y, 1)))) ≈ 2.0
end
