
@testitem "DeltaNode - creation with static inputs (simple case) #1" begin
    import ReactiveMP: nodefunction, DeltaMeta, Linearization

    foo(x, y, z) = x * y + z

    out = randomvar()

    x = randomvar()
    y = datavar()
    z = constvar(3.0)

    node = factornode(foo, [(:out, out), (:in, x), (:in, y), (:in, z)], ((1, 2, 3, 4),))
    meta = DeltaMeta(method = Linearization())

    update!(y, 2.0)

    for xval in rand(10)
        @test nodefunction(node, meta, Val(:out))(xval) === foo(xval, 2.0, 3.0)
        @test nodefunction(node)(foo(xval, 2.0, 3.0), xval) === 1
        @test nodefunction(node)(foo(xval, 2.0, 3.0) + 1.0, xval) === 0
    end
end

@testitem "DeltaNode - Creation with static inputs (all permutations) #2" begin
    import ReactiveMP: nodefunction, DeltaMeta, Linearization

    foo1(x, y, z) = x * y + z
    foo2(x, y, z) = x / y - z
    foo3(x, y, z) = x - y * z

    out = randomvar()

    for vals in [rand(Float64, 3) for _ in 1:10], foo in (foo1, foo2, foo3)

        # In this test we attempt to create a lot of possible combinations 
        # of random, data and const inputs to the delta node
        create_interfaces(i) = ((:in, randomvar()), (:in, datavar()), (:in, constvar(vals[i])))

        for x in create_interfaces(1), y in create_interfaces(2), z in create_interfaces(3)
            in_interfaces = [x, y, z]

            rpos = findall(i -> i isa Tuple{Symbol, RandomVariable}, in_interfaces)
            node = factornode(foo, [(:out, out), in_interfaces...], ((1, 2, 3, 4),))
            meta = DeltaMeta(method = Linearization())

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
