@testitem "Callbacks handler should do absolutely nothing if no handler exists" begin
    import ReactiveMP: invoke_callback

    args = (1, "Hello", [1, 2, 3], [1;;])
    callback_handler = nothing

    function bar(args)
        invoke_callback(callback_handler, Val(:my_event), args)
        return nothing
    end

    bar(args)

    @test bar(args) === nothing
    @test @allocated(bar(args)) === 0
end

@testitem "It should be possible to define custom callback handlers" begin
    import ReactiveMP: invoke_callback

    struct MyCallbackHandler
        events
    end

    function ReactiveMP.invoke_callback(handler::MyCallbackHandler, ::Val{E}, args...) where {E}
        push!(handler.events, (event = E, args = args))
        return nothing
    end

    handler = MyCallbackHandler([])

    @test invoke_callback(handler, Val{:event1}(), 1, 1) === nothing
    @test invoke_callback(handler, Val{:event2}(), 2, 3) === nothing

    @test length(handler.events) === 2
    @test handler.events[1].event === :event1
    @test handler.events[1].args === (1, 1)
    @test handler.events[2].event === :event2
    @test handler.events[2].args === (2, 3)

    @test_throws MethodError invoke_callback(handler, "unsupported type of event", 1, 2)
end

@testitem "NamedTuple should be a supported event handler" begin
    import ReactiveMP: invoke_callback

    callback_handler = (sum_event = (args...) -> sum(args), prod_event = (args...) -> prod(args))

    @test @inferred(invoke_callback(callback_handler, Val{:sum_event}(), 1, 2)) == 3
    @test @inferred(invoke_callback(callback_handler, Val{:sum_event}(), 1, 2, 3)) == 6
    @test @inferred(invoke_callback(callback_handler, Val{:prod_event}(), 1, 2)) == 2
    @test @inferred(invoke_callback(callback_handler, Val{:prod_event}(), 1, 2, 5)) == 10
    @test @inferred(invoke_callback(callback_handler, Val{:other_event}(), 1, 2, 3)) === nothing
end
