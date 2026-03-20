@testmodule CallbacksTestUtils begin
    import ReactiveMP: Event

    struct CustomEvent{E, D} <: Event{E}
        data::D
    end

    function CustomEvent(E::Symbol, args...)
        return CustomEvent{E, typeof(args)}(args)
    end

    export CustomEvent
end

@testitem "Callbacks handler should do absolutely nothing if no handler exists" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback, Event

    # We use here a type stable structure to achieve 0 allocations
    struct MyCustomEvent{T} <: Event{:my_custom_event}
        a::Int
        b::T
        c::Vector{Int}
        d::Matrix{Int}
    end

    event = MyCustomEvent(1, "Hello", [1, 2, 3], [1;;])
    callback_handler = nothing

    function bar(callback_handler, event)
        invoke_callback(callback_handler, event)
        return nothing
    end

    bar(callback_handler, event)

    @test @inferred(bar(callback_handler, event)) === nothing
    @test @allocated(bar(callback_handler, event)) === 0

    function bar2(callback_handler)
        invoke_callback(callback_handler, CustomEvent(:event1, 1, 2, "asd", [2]))
        return 1 + 2
    end

    @test @inferred(bar2(callback_handler)) === 3
    @test @allocated(bar2(callback_handler)) === 0
end

@testitem "event_name should work on both types and instances" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: event_name, Event

    struct MyEvent <: Event{:my_event}
        value::Int
    end

    # Works on types
    @test event_name(MyEvent) === :my_event
    @test event_name(CustomEvent{:foo, Tuple{Int}}) === :foo

    # Works on instances
    @test event_name(MyEvent(42)) === :my_event
    @test event_name(CustomEvent(:bar, 1, 2)) === :bar

    # Works on built-in event types
    @test event_name(ReactiveMP.BeforeMessageRuleCallEvent) === :before_message_rule_call
    @test event_name(ReactiveMP.AfterProductOfTwoMessagesEvent) === :after_product_of_two_messages
end

@testitem "It should be possible to define custom callback handlers" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback, Event

    struct MyCallbackHandler
        events
    end

    function ReactiveMP.invoke_callback(
        handler::MyCallbackHandler, event::Event{E}
    ) where {E}
        push!(handler.events, (event = E, data = event.data))
        return nothing
    end

    handler = MyCallbackHandler([])

    @test invoke_callback(handler, CustomEvent(:event1, 1, 1)) === nothing
    @test invoke_callback(handler, CustomEvent(:event2, 2, 3)) === nothing

    @test length(handler.events) === 2
    @test handler.events[1].event === :event1
    @test handler.events[1].data === (1, 1)
    @test handler.events[2].event === :event2
    @test handler.events[2].data === (2, 3)

    @test_throws MethodError invoke_callback(
        handler, "unsupported type of event"
    )
end

@testitem "invoke_callback error hint for forgotten trailing comma in NamedTuple" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback

    # This simulates the common mistake: `(before_product_of_messages = fn)` without trailing comma.
    # Julia parses this as a plain assignment, so `callbacks` becomes just `fn` (a Function).
    callbacks = (before_product_of_messages = (event) -> nothing)

    # Verify that Julia indeed parsed this as a Function, not a NamedTuple
    @test callbacks isa Function
    @test !(callbacks isa NamedTuple)

    err = try
        invoke_callback(callbacks, CustomEvent(:before_product_of_messages))
    catch e
        e
    end
    @test err isa MethodError

    # Check that the error hint mentions both possible causes
    hint_message = sprint(showerror, err)
    @test occursin("invoke_callback", hint_message)
    @test occursin("trailing comma", hint_message)
end

@testitem "invoke_callback error hint for custom handler with missing method" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback

    # Custom handler that only implements invoke_callback for :event1 but not :event2
    struct IncompleteHandler end

    ReactiveMP.invoke_callback(::IncompleteHandler, ::CustomEvent{:event1}) =
        nothing

    handler = IncompleteHandler()

    # :event1 works fine
    @test invoke_callback(handler, CustomEvent(:event1)) === nothing

    # :event2 is not implemented — should hit MethodError with a helpful hint
    err = try
        invoke_callback(handler, CustomEvent(:event2))
    catch e
        e
    end
    @test err isa MethodError

    hint_message = sprint(showerror, err)
    @test occursin(
        r"ReactiveMP\.invoke_callback\(::.*IncompleteHandler, event::Event\{:event2\}\) = \.\.\.",
        hint_message,
    )
    @test occursin(
        "You meant to pass a `NamedTuple` as the callbacks handler but forgot the trailing comma.",
        hint_message,
    )
end

@testitem "NamedTuple should be a supported event handler" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback

    callback_handler = (
        sum_event = (event) -> sum(event.data),
        prod_event = (event) -> prod(event.data),
    )

    @test @inferred(
        invoke_callback(callback_handler, CustomEvent(:sum_event, 1, 2))
    ) == 3
    @test @inferred(
        invoke_callback(callback_handler, CustomEvent(:prod_event, 1, 2, 3))
    ) == 6
    @test @inferred(
        invoke_callback(callback_handler, CustomEvent(:prod_event, 1, 2))
    ) == 2
    @test @inferred(
        invoke_callback(callback_handler, CustomEvent(:prod_event, 2, 5))
    ) == 10
    @test @inferred(
        invoke_callback(callback_handler, CustomEvent(:other_event, 1, 2, 3))
    ) === nothing
end

@testitem "Dict{Symbol} should be a supported event handler" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback

    callback_handler = Dict{Symbol, Any}(
        :sum_event => (event) -> sum(event.data),
        :prod_event => (event) -> prod(event.data),
    )

    @test invoke_callback(callback_handler, CustomEvent(:sum_event, 1, 2)) == 3
    @test invoke_callback(callback_handler, CustomEvent(:sum_event, 1, 2, 3)) ==
        6
    @test invoke_callback(callback_handler, CustomEvent(:prod_event, 1, 2)) == 2
    @test invoke_callback(
        callback_handler, CustomEvent(:prod_event, 1, 2, 5)
    ) == 10
    @test invoke_callback(
        callback_handler, CustomEvent(:other_event, 1, 2, 3)
    ) === nothing
end

@testitem "It should be possible to merge callback handlers" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback, merge_callbacks, Event

    # listens to event 1 and event 2
    handler1_events = []
    callback_handler1 = (
        event1 = (event) -> push!(handler1_events, :event1),
        event2 = (event) -> push!(handler1_events, :event2),
    )

    # listens to event3 and event 2
    handler2_events = []
    callback_handler2 = (
        event3 = (event) -> push!(handler2_events, :event3),
        event2 = (event) -> push!(handler2_events, :event2),
    )

    # only listens to event 2
    struct MyCustomHandler
        events
    end

    ReactiveMP.invoke_callback(::MyCustomHandler, ::Event) = nothing
    ReactiveMP.invoke_callback(
        handler::MyCustomHandler, ::CustomEvent{:event2}
    ) = push!(handler.events, :event2)

    custom_handler = MyCustomHandler([])

    merged_handler = merge_callbacks(
        callback_handler1, callback_handler2, custom_handler
    )

    for i in 1:5
        invoke_callback(merged_handler, CustomEvent(:event1, 1, 1))
        invoke_callback(merged_handler, CustomEvent(:event2, "hello"))
        invoke_callback(merged_handler, CustomEvent(:event3, 3.0))
    end

    @test length(handler1_events) == 10
    @test Set(handler1_events) == Set([:event1, :event2])
    @test length(handler2_events) == 10
    @test Set(handler2_events) == Set([:event3, :event2])
    @test length(custom_handler.events) == 5
    @test Set(custom_handler.events) == Set([:event2])
end

@testitem "It should be possible to reduce the result of the merged callback handlers" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback, merge_callbacks

    callback_handler1 = (event1 = (event) -> event.data[1] + event.data[2],)
    callback_handler2 = (event1 = (event) -> event.data[1] * event.data[2],)

    merged_handler1 = merge_callbacks(callback_handler1, callback_handler2)

    @test @inferred(
        invoke_callback(merged_handler1, CustomEvent(:event1, 2, 3))
    ) === (5, 6)

    merged_handler2 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = +
    )

    @test @inferred(
        invoke_callback(merged_handler2, CustomEvent(:event1, 4, 5))
    ) === 29

    merged_handler3 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = *
    )

    @test @inferred(
        invoke_callback(merged_handler3, CustomEvent(:event1, 1.0, 2.0))
    ) === 6.0
end

@testitem "It should be possible to use different reduce functions for different events" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback, merge_callbacks

    callback_handler1 = (
        event1 = (event) -> event.data[1] + event.data[2],
        event2 = (event) -> event.data[1] - event.data[2],
    )
    callback_handler2 = (
        event1 = (event) -> event.data[1] * event.data[2],
        event2 = (event) -> event.data[1] / event.data[2],
    )

    merged_handler1 = merge_callbacks(callback_handler1, callback_handler2)

    @test @inferred(
        invoke_callback(merged_handler1, CustomEvent(:event1, 2, 3))
    ) === (5, 6)
    @test @inferred(
        invoke_callback(merged_handler1, CustomEvent(:event2, 3, 4))
    ) === (-1, 3 / 4)

    merged_handler2 = merge_callbacks(
        callback_handler1,
        callback_handler2;
        reduce_fn = (event1 = +, event2 = *),
    )

    @test @inferred(
        invoke_callback(merged_handler2, CustomEvent(:event1, 4, 5))
    ) === 29
    @test @inferred(
        invoke_callback(merged_handler2, CustomEvent(:event2, 4, 5))
    ) === -4 / 5

    merged_handler3 = merge_callbacks(
        callback_handler1,
        callback_handler2;
        reduce_fn = (event1 = *, event2 = +),
    )

    @test @inferred(
        invoke_callback(merged_handler3, CustomEvent(:event1, 1.0, 2.0))
    ) === 6.0
    @test @inferred(
        invoke_callback(merged_handler3, CustomEvent(:event2, 1.0, 2.0))
    ) === -1.0 + 1.0 / 2.0

    merged_handler4 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = (event1 = -,)
    )

    @test @inferred(
        invoke_callback(merged_handler4, CustomEvent(:event1, 1.0, 2.0))
    ) === 1.0
    @test @inferred(
        invoke_callback(merged_handler4, CustomEvent(:event2, 1.0, 2.0))
    ) === (-1.0, 1.0 / 2.0)

    merged_handler5 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = (event2 = /,)
    )

    @test @inferred(
        invoke_callback(merged_handler5, CustomEvent(:event1, 1.0, 2.0))
    ) === (3.0, 2.0)
    @test @inferred(
        invoke_callback(merged_handler5, CustomEvent(:event2, 1.0, 2.0))
    ) === -1.0 / (1.0 / 2.0)
end
