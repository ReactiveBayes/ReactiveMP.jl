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

    function ReactiveMP.invoke_callback(
        handler::MyCallbackHandler, ::Val{E}, args...
    ) where {E}
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

    @test_throws MethodError invoke_callback(
        handler, "unsupported type of event", 1, 2
    )
end

@testitem "invoke_callback error hint for forgotten trailing comma in NamedTuple" begin
    import ReactiveMP: invoke_callback

    # This simulates the common mistake: `(before_product_of_messages = fn)` without trailing comma.
    # Julia parses this as a plain assignment, so `callbacks` becomes just `fn` (a Function).
    callbacks = (before_product_of_messages = (args...) -> nothing)

    # Verify that Julia indeed parsed this as a Function, not a NamedTuple
    @test callbacks isa Function
    @test !(callbacks isa NamedTuple)

    err = try
        invoke_callback(callbacks, Val(:before_product_of_messages), 1, 2)
    catch e
        e
    end
    @test err isa MethodError

    # Check that the error hint mentions both possible causes
    hint_message = sprint(showerror, err)
    @test occursin("invoke_callback", hint_message)
    @test occursin("trailing comma", hint_message)
end

@testitem "invoke_callback error hint for custom handler with missing method" begin
    import ReactiveMP: invoke_callback

    # Custom handler that only implements invoke_callback for :event1 but not :event2
    struct IncompleteHandler end

    ReactiveMP.invoke_callback(::IncompleteHandler, ::Val{:event1}, args...) =
        nothing

    handler = IncompleteHandler()

    # :event1 works fine
    @test invoke_callback(handler, Val(:event1), 1, 2) === nothing

    # :event2 is not implemented — should hit MethodError with a helpful hint
    err = try
        invoke_callback(handler, Val(:event2), 1, 2)
    catch e
        e
    end
    @test err isa MethodError

    hint_message = sprint(showerror, err)
    @test occursin(
        r"ReactiveMP\.invoke_callback\(::.*IncompleteHandler, ::Val\{:event2\}, args\.\.\.\) = \.\.\.",
        hint_message,
    )
    @test occursin(
        "You meant to pass a `NamedTuple` as the callbacks handler but forgot the trailing comma.",
        hint_message,
    )
end

@testitem "NamedTuple should be a supported event handler" begin
    import ReactiveMP: invoke_callback

    callback_handler = (
        sum_event = (args...) -> sum(args), prod_event = (args...) -> prod(args)
    )

    @test @inferred(
        invoke_callback(callback_handler, Val{:sum_event}(), 1, 2)
    ) == 3
    @test @inferred(
        invoke_callback(callback_handler, Val{:sum_event}(), 1, 2, 3)
    ) == 6
    @test @inferred(
        invoke_callback(callback_handler, Val{:prod_event}(), 1, 2)
    ) == 2
    @test @inferred(
        invoke_callback(callback_handler, Val{:prod_event}(), 1, 2, 5)
    ) == 10
    @test @inferred(
        invoke_callback(callback_handler, Val{:other_event}(), 1, 2, 3)
    ) === nothing
end

@testitem "It should be possible to merge callback handlers" begin
    import ReactiveMP: invoke_callback, merge_callbacks

    # listens to event 1 and event 2
    handler1_events = []
    callback_handler1 = (
        event1 = (args...) -> push!(handler1_events, :event1),
        event2 = (args...) -> push!(handler1_events, :event2),
    )

    # listens to event3 and event 2
    handler2_events = []
    callback_handler2 = (
        event3 = (args...) -> push!(handler2_events, :event3),
        event2 = (args...) -> push!(handler2_events, :event2),
    )

    # only listens to event 2
    struct MyCustomHandler
        events
    end

    ReactiveMP.invoke_callback(::MyCustomHandler, event, args...) = nothing
    ReactiveMP.invoke_callback(
        handler::MyCustomHandler, event::Val{:event2}, args...
    ) = push!(handler.events, :event2)

    custom_handler = MyCustomHandler([])

    merged_handler = merge_callbacks(
        callback_handler1, callback_handler2, custom_handler
    )

    for i in 1:5
        invoke_callback(merged_handler, Val(:event1), 1, 1)
        invoke_callback(merged_handler, Val(:event2), "hello")
        invoke_callback(merged_handler, Val(:event3), 3.0)
    end

    @test length(handler1_events) == 10
    @test Set(handler1_events) == Set([:event1, :event2])
    @test length(handler2_events) == 10
    @test Set(handler2_events) == Set([:event3, :event2])
    @test length(custom_handler.events) == 5
    @test Set(custom_handler.events) == Set([:event2])
end

@testitem "It should be possible to reduce the result of the merged callback handlers" begin
    import ReactiveMP: invoke_callback, merge_callbacks

    callback_handler1 = (event1 = (a, b) -> a + b,)
    callback_handler2 = (event1 = (a, b) -> a * b,)

    merged_handler1 = merge_callbacks(callback_handler1, callback_handler2)

    @test @inferred(invoke_callback(merged_handler1, Val(:event1), 2, 3)) ===
        (5, 6)

    merged_handler2 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = +
    )

    @test @inferred(invoke_callback(merged_handler2, Val(:event1), 4, 5)) === 29

    merged_handler3 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = *
    )

    @test @inferred(
        invoke_callback(merged_handler3, Val(:event1), 1.0, 2.0)
    ) === 6.0
end

@testitem "It should be possible to use different reduce functions for different events" begin
    import ReactiveMP: invoke_callback, merge_callbacks

    callback_handler1 = (event1 = (a, b) -> a + b, event2 = (a, b) -> a - b)
    callback_handler2 = (event1 = (a, b) -> a * b, event2 = (a, b) -> a / b)

    merged_handler1 = merge_callbacks(callback_handler1, callback_handler2)

    @test @inferred(invoke_callback(merged_handler1, Val(:event1), 2, 3)) ===
        (5, 6)
    @test @inferred(invoke_callback(merged_handler1, Val(:event2), 3, 4)) ===
        (-1, 3 / 4)

    merged_handler2 = merge_callbacks(
        callback_handler1,
        callback_handler2;
        reduce_fn = (event1 = +, event2 = *),
    )

    @test @inferred(invoke_callback(merged_handler2, Val(:event1), 4, 5)) === 29
    @test @inferred(invoke_callback(merged_handler2, Val(:event2), 4, 5)) ===
        -4 / 5

    merged_handler3 = merge_callbacks(
        callback_handler1,
        callback_handler2;
        reduce_fn = (event1 = *, event2 = +),
    )

    @test @inferred(
        invoke_callback(merged_handler3, Val(:event1), 1.0, 2.0)
    ) === 6.0
    @test @inferred(
        invoke_callback(merged_handler3, Val(:event2), 1.0, 2.0)
    ) === -1.0 + 1.0 / 2.0

    merged_handler4 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = (event1 = -,)
    )

    @test @inferred(
        invoke_callback(merged_handler4, Val(:event1), 1.0, 2.0)
    ) === 1.0
    @test @inferred(
        invoke_callback(merged_handler4, Val(:event2), 1.0, 2.0)
    ) === (-1.0, 1.0 / 2.0)

    merged_handler5 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = (event2 = /,)
    )

    @test @inferred(
        invoke_callback(merged_handler5, Val(:event1), 1.0, 2.0)
    ) === (3.0, 2.0)
    @test @inferred(
        invoke_callback(merged_handler5, Val(:event2), 1.0, 2.0)
    ) === -1.0 / (1.0 / 2.0)
end
