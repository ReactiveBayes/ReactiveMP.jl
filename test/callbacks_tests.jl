@testitem "Callbacks handler should do absolutely nothing if no handler exists" begin
    import ReactiveMP: invoke_callback

    Base.@kwdef struct MyNoopEventData{X}
        x::X
    end

    data = MyNoopEventData(x = 42)
    callback_handler = nothing

    function bar(data)
        invoke_callback(callback_handler, Val{:my_noop_event}(), data)
        return nothing
    end

    bar(data)

    @test bar(data) === nothing
    @test @allocated(bar(data)) === 0
end

@testitem "It should be possible to define custom callback handlers" begin
    import ReactiveMP: invoke_callback

    Base.@kwdef struct CustomEvent1Data{A, B}
        a::A
        b::B
    end

    Base.@kwdef struct CustomEvent2Data{A, B}
        a::A
        b::B
    end

    struct MyCallbackHandler
        events
    end

    function ReactiveMP.invoke_callback(handler::MyCallbackHandler, ::Val{E}, data) where {E}
        push!(handler.events, (event = E, data = data))
        return nothing
    end

    handler = MyCallbackHandler([])

    @test invoke_callback(handler, Val{:event1}(), CustomEvent1Data(a = 1, b = 2)) === nothing
    @test invoke_callback(handler, Val{:event2}(), CustomEvent2Data(a = 3, b = 4)) === nothing

    @test length(handler.events) === 2
    @test handler.events[1].event === :event1
    @test handler.events[1].data.a === 1
    @test handler.events[1].data.b === 2
    @test handler.events[2].event === :event2
    @test handler.events[2].data.a === 3
    @test handler.events[2].data.b === 4
end

@testitem "invoke_callback error hint for forgotten trailing comma in NamedTuple" begin
    import ReactiveMP: invoke_callback

    Base.@kwdef struct SomeEventData{X}
        x::X
    end

    # This simulates the common mistake: `(before_product_of_messages = fn)` without trailing comma.
    # Julia parses this as a plain assignment, so `callbacks` becomes just `fn` (a Function).
    callbacks = (before_product_of_messages = (data) -> nothing)

    # Verify that Julia indeed parsed this as a Function, not a NamedTuple
    @test callbacks isa Function
    @test !(callbacks isa NamedTuple)

    err = try
        invoke_callback(callbacks, Val(:before_product_of_messages), SomeEventData(x = 1))
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

    Base.@kwdef struct Ev1Data{X}
        x::X
    end

    Base.@kwdef struct Ev2Data{X}
        x::X
    end

    # Custom handler that only implements invoke_callback for :event1 but not :event2
    struct IncompleteHandler end

    ReactiveMP.invoke_callback(::IncompleteHandler, ::Val{:event1}, data::Ev1Data) = nothing

    handler = IncompleteHandler()

    # :event1 works fine
    @test invoke_callback(handler, Val(:event1), Ev1Data(x = 1)) === nothing

    # :event2 is not implemented — should hit MethodError with a helpful hint
    err = try
        invoke_callback(handler, Val(:event2), Ev2Data(x = 2))
    catch e
        e
    end
    @test err isa MethodError

    hint_message = sprint(showerror, err)
    @test occursin("invoke_callback", hint_message)
    @test occursin(
        "You meant to pass a `NamedTuple` as the callbacks handler but forgot the trailing comma.",
        hint_message,
    )
end

@testitem "NamedTuple should be a supported event handler" begin
    import ReactiveMP: invoke_callback

    Base.@kwdef struct SumEventData{A, B}
        a::A
        b::B
    end

    Base.@kwdef struct ProdEventData{A, B}
        a::A
        b::B
    end

    Base.@kwdef struct OtherEventData{X}
        x::X
    end

    callback_handler = (
        sum_event = (data) -> data.a + data.b,
        prod_event = (data) -> data.a * data.b,
    )

    @test @inferred(
        invoke_callback(callback_handler, Val{:sum_event}(), SumEventData(a = 1, b = 2))
    ) == 3
    @test @inferred(
        invoke_callback(callback_handler, Val{:sum_event}(), SumEventData(a = 1, b = 5))
    ) == 6
    @test @inferred(
        invoke_callback(callback_handler, Val{:prod_event}(), ProdEventData(a = 1, b = 2))
    ) == 2
    @test @inferred(
        invoke_callback(callback_handler, Val{:prod_event}(), ProdEventData(a = 2, b = 5))
    ) == 10
    @test @inferred(
        invoke_callback(callback_handler, Val{:other_event}(), OtherEventData(x = 1))
    ) === nothing
end

@testitem "Dict{Symbol} should be a supported event handler" begin
    import ReactiveMP: invoke_callback

    Base.@kwdef struct DictSumEventData{A, B}
        a::A
        b::B
    end

    Base.@kwdef struct DictProdEventData{A, B}
        a::A
        b::B
    end

    Base.@kwdef struct DictOtherEventData{X}
        x::X
    end

    callback_handler = Dict{Symbol, Any}(
        :sum_event => (data) -> data.a + data.b,
        :prod_event => (data) -> data.a * data.b,
    )

    @test invoke_callback(callback_handler, Val{:sum_event}(), DictSumEventData(a = 1, b = 2)) == 3
    @test invoke_callback(callback_handler, Val{:sum_event}(), DictSumEventData(a = 1, b = 5)) == 6
    @test invoke_callback(callback_handler, Val{:prod_event}(), DictProdEventData(a = 1, b = 2)) == 2
    @test invoke_callback(callback_handler, Val{:prod_event}(), DictProdEventData(a = 2, b = 5)) == 10
    @test invoke_callback(callback_handler, Val{:other_event}(), DictOtherEventData(x = 1)) === nothing
end

@testitem "It should be possible to merge callback handlers" begin
    import ReactiveMP: invoke_callback, merge_callbacks

    Base.@kwdef struct MergeEvent1Data{X}
        x::X
    end

    Base.@kwdef struct MergeEvent2Data{X}
        x::X
    end

    Base.@kwdef struct MergeEvent3Data{X}
        x::X
    end

    # listens to event 1 and event 2
    handler1_events = []
    callback_handler1 = (
        event1 = (data) -> push!(handler1_events, :event1),
        event2 = (data) -> push!(handler1_events, :event2),
    )

    # listens to event3 and event 2
    handler2_events = []
    callback_handler2 = (
        event3 = (data) -> push!(handler2_events, :event3),
        event2 = (data) -> push!(handler2_events, :event2),
    )

    # only listens to event 2
    struct MyCustomMergeHandler
        events
    end

    ReactiveMP.invoke_callback(::MyCustomMergeHandler, ::Val, data) = nothing
    ReactiveMP.invoke_callback(
        handler::MyCustomMergeHandler, ::Val{:event2}, data::MergeEvent2Data
    ) = push!(handler.events, :event2)

    custom_handler = MyCustomMergeHandler([])

    merged_handler = merge_callbacks(
        callback_handler1, callback_handler2, custom_handler
    )

    for i in 1:5
        invoke_callback(merged_handler, Val(:event1), MergeEvent1Data(x = i))
        invoke_callback(merged_handler, Val(:event2), MergeEvent2Data(x = i))
        invoke_callback(merged_handler, Val(:event3), MergeEvent3Data(x = i))
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

    Base.@kwdef struct ReduceEvent1Data{A, B}
        a::A
        b::B
    end

    callback_handler1 = (event1 = (data) -> data.a + data.b,)
    callback_handler2 = (event1 = (data) -> data.a * data.b,)

    merged_handler1 = merge_callbacks(callback_handler1, callback_handler2)

    @test @inferred(
        invoke_callback(merged_handler1, Val(:event1), ReduceEvent1Data(a = 2, b = 3))
    ) === (5, 6)

    merged_handler2 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = +
    )

    @test @inferred(
        invoke_callback(merged_handler2, Val(:event1), ReduceEvent1Data(a = 4, b = 5))
    ) === 29

    merged_handler3 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = *
    )

    @test @inferred(
        invoke_callback(merged_handler3, Val(:event1), ReduceEvent1Data(a = 1.0, b = 2.0))
    ) === 6.0
end

@testitem "It should be possible to use different reduce functions for different events" begin
    import ReactiveMP: invoke_callback, merge_callbacks

    Base.@kwdef struct PerEvA{A, B}
        a::A
        b::B
    end

    Base.@kwdef struct PerEvB{A, B}
        a::A
        b::B
    end

    callback_handler1 = (event1 = (data) -> data.a + data.b, event2 = (data) -> data.a - data.b)
    callback_handler2 = (event1 = (data) -> data.a * data.b, event2 = (data) -> data.a / data.b)

    merged_handler1 = merge_callbacks(callback_handler1, callback_handler2)

    @test @inferred(invoke_callback(merged_handler1, Val(:event1), PerEvA(a = 2, b = 3))) ===
        (5, 6)
    @test @inferred(invoke_callback(merged_handler1, Val(:event2), PerEvB(a = 3, b = 4))) ===
        (-1, 3 / 4)

    merged_handler2 = merge_callbacks(
        callback_handler1,
        callback_handler2;
        reduce_fn = (event1 = +, event2 = *),
    )

    @test @inferred(invoke_callback(merged_handler2, Val(:event1), PerEvA(a = 4, b = 5))) === 29
    @test @inferred(invoke_callback(merged_handler2, Val(:event2), PerEvB(a = 4, b = 5))) ===
        -4 / 5

    merged_handler3 = merge_callbacks(
        callback_handler1,
        callback_handler2;
        reduce_fn = (event1 = *, event2 = +),
    )

    @test @inferred(
        invoke_callback(merged_handler3, Val(:event1), PerEvA(a = 1.0, b = 2.0))
    ) === 6.0
    @test @inferred(
        invoke_callback(merged_handler3, Val(:event2), PerEvB(a = 1.0, b = 2.0))
    ) === -1.0 + 1.0 / 2.0

    merged_handler4 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = (event1 = -,)
    )

    @test @inferred(
        invoke_callback(merged_handler4, Val(:event1), PerEvA(a = 1.0, b = 2.0))
    ) === 1.0
    @test @inferred(
        invoke_callback(merged_handler4, Val(:event2), PerEvB(a = 1.0, b = 2.0))
    ) === (-1.0, 1.0 / 2.0)

    merged_handler5 = merge_callbacks(
        callback_handler1, callback_handler2; reduce_fn = (event2 = /,)
    )

    @test @inferred(
        invoke_callback(merged_handler5, Val(:event1), PerEvA(a = 1.0, b = 2.0))
    ) === (3.0, 2.0)
    @test @inferred(
        invoke_callback(merged_handler5, Val(:event2), PerEvB(a = 1.0, b = 2.0))
    ) === -1.0 / (1.0 / 2.0)
end
