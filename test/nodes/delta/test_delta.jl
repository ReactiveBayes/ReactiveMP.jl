module DeltaNodeTest

using Test, ReactiveMP, Random

@testset "DeltaNode" begin

    @testset "Creation with static inputs (simple case) #1" begin
        import ReactiveMP: nodefunction, FactorNodeCreationOptions

        foo(x, y, z) = x * y + z

        out = randomvar(:out)
        
        x = randomvar(:x)
        y = datavar(:y, Float64)
        z = constvar(:z, 3.0)

        node = make_node(foo, FactorNodeCreationOptions(nothing, Linearization(), nothing), out, x, y, z)

        update!(y, 2.0)

        for xval in rand(10)
            @test nodefunction(node, Val(:out))(xval) === foo(xval, 2.0, 3.0)
            @test nodefunction(node)(foo(xval, 2.0, 3.0), xval) === 1
            @test nodefunction(node)(foo(xval, 2.0, 3.0) + 1.0, xval) === 0
        end

    end

    @testset "Creation with static inputs (all permutations) #2" begin
        import ReactiveMP: nodefunction, FactorNodeCreationOptions

        foo1(x, y, z) = x * y + z
        foo2(x, y, z) = x / y - z
        foo3(x, y, z) = x - y * z

        out = randomvar(:out)
        opt = FactorNodeCreationOptions(nothing, Linearization(), nothing)

        for vals in [ rand(Float64, 3) for _ in 1:10 ], foo in (foo1, foo2, foo3)

            # In this test we attempt to create a lot of possible combinations 
            # of random, data and const inputs to the delta node
            create_interfaces(i) = (randomvar(:x), datavar(:y, Float64), constvar(:z, vals[i]))

            for x in create_interfaces(1), y in create_interfaces(2), z in create_interfaces(3)
                interfaces = [ x, y, z ]

                rpos = findall(i -> i isa RandomVariable, interfaces)
                node = make_node(foo, opt, out, interfaces...)

                # data variable inputs require an actual update
                foreach(enumerate(interfaces)) do (i, interface)
                    if interface isa DataVariable
                        update!(interface, vals[i])
                    end
                end

                @test nodefunction(node, Val(:out))(vals[rpos]...) === foo(vals...)
                @test nodefunction(node)(foo(vals...), vals[rpos]...) === 1
                @test nodefunction(node)(foo(vals...) + 1, vals[rpos]...) === 0
            end
            
        end

    end

end

end