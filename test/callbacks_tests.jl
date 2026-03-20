@testmodule CallbacksTestUtils begin
    import ReactiveMP: Event

    struct CustomEvent{E, D} <: Event{E}
        data::D
    end

    function CustomEvent(E::Symbol, args...)
        return CustomEvent{E, typeof(args)}(args)
    end

    mutable struct MutableCustomEvent{E, D} <: Event{E}
        data::D
        state::Any
    end

    function MutableCustomEvent(E::Symbol, args...; state = nothing)
        return MutableCustomEvent{E, typeof(args)}(args, state)
    end

    export CustomEvent, MutableCustomEvent
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
        invoke_callback(
            callback_handler, CustomEvent(:event1, 1, 2, "asd", [2])
        )
        return 1 + 2
    end

    bar2(callback_handler)

    @test @inferred(bar2(callback_handler)) === 3
    @test @allocated(bar2(callback_handler)) === 0

    mutable struct EventWithTypeStableState <:
                   Event{:event_with_typestable_state}
        internal_state::Bool
    end

    function bar3(callback_handler)
        event = EventWithTypeStableState(true)
        invoke_callback(callback_handler, event)
        return event.internal_state
    end

    bar3(callback_handler)

    @test @inferred(bar3(callback_handler)) === true
    @test @allocated(bar3(callback_handler)) === 0

    mutable struct EventWithTypeUnstableState <:
                   Event{:event_with_typeunstable_state}
        state
    end

    function bar4(callback_handler)
        event = EventWithTypeUnstableState(nothing)
        invoke_callback(callback_handler, event)
        if event.state === nothing
            return 1
        end
        return [1]
    end

    bar4(callback_handler)

    @test @allocated(bar4(callback_handler)) === 0
end

@testitem "invoke_callback should return the event" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback

    event = CustomEvent(:event1, 1, 2)

    # nothing handler returns the event
    @test invoke_callback(nothing, event) === event

    # NamedTuple handler returns the event
    callbacks = (event1 = (e) -> nothing,)
    @test invoke_callback(callbacks, event) === event

    # Dict handler returns the event
    dict_callbacks = Dict{Symbol, Any}(:event1 => (e) -> nothing)
    @test invoke_callback(dict_callbacks, event) === event

    # Unmatched event still returns the event
    @test invoke_callback(callbacks, CustomEvent(:other, 1)) === CustomEvent(:other, 1)
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
    @test event_name(ReactiveMP.BeforeMessageRuleCallEvent) ===
        :before_message_rule_call
    @test event_name(ReactiveMP.AfterProductOfTwoMessagesEvent) ===
        :after_product_of_two_messages
end

@testitem "It should be possible to define custom callback handlers via handle_event" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback, handle_event, Event

    struct MyCallbackHandler
        events
    end

    function ReactiveMP.handle_event(
        handler::MyCallbackHandler, event::Event{E}
    ) where {E}
        push!(handler.events, (event = E, data = event.data))
        return nothing
    end

    handler = MyCallbackHandler([])

    @test invoke_callback(handler, CustomEvent(:event1, 1, 1)) isa
        CustomEvent{:event1}
    @test invoke_callback(handler, CustomEvent(:event2, 2, 3)) isa
        CustomEvent{:event2}

    @test length(handler.events) === 2
    @test handler.events[1].event === :event1
    @test handler.events[1].data === (1, 1)
    @test handler.events[2].event === :event2
    @test handler.events[2].data === (2, 3)

    @test_throws MethodError invoke_callback(
        handler, "unsupported type of event"
    )
end

@testitem "Custom callback handler can mutate event state" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback, handle_event, Event

    struct StateModifyingHandler end

    function ReactiveMP.handle_event(
        ::StateModifyingHandler, event::MutableCustomEvent{:my_event}
    )
        event.state = :modified
        return nothing
    end

    handler = StateModifyingHandler()
    event = MutableCustomEvent(:my_event, 1, 2; state = nothing)

    @test event.state === nothing
    returned_event = invoke_callback(handler, event)
    @test returned_event === event
    @test event.state === :modified
end

@testitem "NamedTuple callback can mutate event state" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback

    callbacks = (
        my_event = (event) -> begin
            event.state = sum(event.data)
        end,
    )

    event = MutableCustomEvent(:my_event, 3, 4; state = nothing)
    returned_event = invoke_callback(callbacks, event)

    @test returned_event === event
    @test event.state === 7
end

@testitem "Dict callback can mutate event state" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback

    callbacks = Dict{Symbol, Any}(
        :my_event => (event) -> begin
            event.state = prod(event.data)
        end,
    )

    event = MutableCustomEvent(:my_event, 3, 5; state = nothing)
    returned_event = invoke_callback(callbacks, event)

    @test returned_event === event
    @test event.state === 15
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
    @test occursin("handle_event", hint_message)
    @test occursin("trailing comma", hint_message)
end

@testitem "invoke_callback error hint for custom handler with missing method" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback

    # Custom handler that only implements handle_event for :event1 but not :event2
    struct IncompleteHandler end

    ReactiveMP.handle_event(::IncompleteHandler, ::CustomEvent{:event1}) =
        nothing

    handler = IncompleteHandler()

    # :event1 works fine
    @test invoke_callback(handler, CustomEvent(:event1)) isa
        CustomEvent{:event1}

    # :event2 is not implemented — should hit MethodError with a helpful hint
    err = try
        invoke_callback(handler, CustomEvent(:event2))
    catch e
        e
    end
    @test err isa MethodError

    hint_message = sprint(showerror, err)
    @test occursin(
        r"ReactiveMP\.handle_event\(::.*IncompleteHandler, event::Event\{:event2\}\) = \.\.\.",
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
        sum_event = (event) -> nothing,
        prod_event = (event) -> nothing,
    )

    @test invoke_callback(callback_handler, CustomEvent(:sum_event, 1, 2)) isa
        CustomEvent{:sum_event}
    @test invoke_callback(
        callback_handler, CustomEvent(:prod_event, 1, 2, 3)
    ) isa CustomEvent{:prod_event}
    @test invoke_callback(
        callback_handler, CustomEvent(:other_event, 1, 2, 3)
    ) isa CustomEvent{:other_event}
end

@testitem "Dict{Symbol} should be a supported event handler" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback

    callback_handler = Dict{Symbol, Any}(
        :sum_event => (event) -> nothing,
        :prod_event => (event) -> nothing,
    )

    @test invoke_callback(callback_handler, CustomEvent(:sum_event, 1, 2)) isa
        CustomEvent{:sum_event}
    @test invoke_callback(callback_handler, CustomEvent(:prod_event, 1, 2)) isa
        CustomEvent{:prod_event}
    @test invoke_callback(
        callback_handler, CustomEvent(:other_event, 1, 2, 3)
    ) === CustomEvent(:other_event, 1, 2, 3)
end

@testitem "It should be possible to merge callback handlers" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback, merge_callbacks, handle_event, Event

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

    ReactiveMP.handle_event(::MyCustomHandler, ::Event) = nothing
    ReactiveMP.handle_event(
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

@testitem "Merged callbacks should return the event" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback, merge_callbacks

    callback_handler1 = (event1 = (event) -> nothing,)
    callback_handler2 = (event1 = (event) -> nothing,)

    merged_handler = merge_callbacks(callback_handler1, callback_handler2)

    event = CustomEvent(:event1, 2, 3)
    @test invoke_callback(merged_handler, event) === event
end

@testitem "Merged callbacks can mutate event state across handlers" setup = [
    CallbacksTestUtils
] begin
    import ReactiveMP: invoke_callback, merge_callbacks

    # First handler sets state to 1
    handler1 = (my_event = (event) -> begin
        event.state = 1
    end,)

    # Second handler increments state
    handler2 = (my_event = (event) -> begin
        event.state += 10
    end,)

    merged_handler = merge_callbacks(handler1, handler2)

    event = MutableCustomEvent(:my_event, 1, 2; state = nothing)
    returned_event = invoke_callback(merged_handler, event)

    @test returned_event === event
    @test event.state === 11
end
