@testitem "Event handler should do absolutely nothing if no handler exists" begin
    import ReactiveMP: broadcast_event

    args = (1, "Hello", [1, 2, 3], [1;;])
    event_handler = nothing

    function bar(args)
        broadcast_event(event_handler, Val(:my_event), args)
        return nothing
    end

    bar(args)

    @test bar(args) === nothing
    @test @allocated(bar(args)) === 0
end

@testitem "It should be possible to define custom event handlers" begin
    import ReactiveMP: broadcast_event, Event, handle_event

    struct MyEventHandler
        events
    end

    function ReactiveMP.handle_event(handler::MyEventHandler, ::Event{E}, args...) where {E}
        push!(handler.events, (event = E, args = args))
        return nothing
    end

    handler = MyEventHandler([])

    @test broadcast_event(handler, Event{:event1}(), 1, 1) === nothing
    @test broadcast_event(handler, Event{:event2}(), 2, 3) === nothing

    @test length(handler.events) === 2
    @test handler.events[1].event === :event1
    @test handler.events[1].args === (1, 1)
    @test handler.events[2].event === :event2
    @test handler.events[2].args === (2, 3)

    @test_throws MethodError broadcast_event(handler, "unsupported type of event", 1, 2)
end

@testitem "NamedTuple should be a supported event handler" begin
    import ReactiveMP: broadcast_event, Event

    event_handler = (sum_event = (args...) -> sum(args), prod_event = (args...) -> prod(args))

    @test @inferred(broadcast_event(event_handler, Event{:sum_event}(), 1, 2)) == 3
    @test @inferred(broadcast_event(event_handler, Event{:sum_event}(), 1, 2, 3)) == 6
    @test @inferred(broadcast_event(event_handler, Event{:prod_event}(), 1, 2)) == 2
    @test @inferred(broadcast_event(event_handler, Event{:prod_event}(), 1, 2, 5)) == 10
    @test @inferred(broadcast_event(event_handler, Event{:other_event}(), 1, 2, 3)) === nothing
end
