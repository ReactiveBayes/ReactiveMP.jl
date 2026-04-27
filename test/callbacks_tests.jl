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
    import ReactiveMP: invoke_callback, Event, generate_span_id
    using UUIDs

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

    if VERSION >= v"1.12.0"
        function bar2(callback_handler)
            invoke_callback(
                callback_handler, CustomEvent(:event1, 1, 2, "asd", [2])
            )
            return 1 + 2
        end

        bar2(callback_handler)

        @test @inferred(bar2(callback_handler)) === 3
        @test @allocated(bar2(callback_handler)) === 0
    end

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

    if VERSION >= v"1.12.0"
        # Test that span_id does not cause allocations
        struct BeforeSuperCoolEvent{I} <: Event{:my_custom_event_243}
            span_id::I
        end
        struct AfterSuperCoolEvent{I} <: Event{:my_custom_event_534}
            result::Float64
            span_id::I
        end

        function bar5(callback_handler, input::Float64)
            span_id = generate_span_id(callback_handler)

            invoke_callback(callback_handler, BeforeSuperCoolEvent(span_id))

            result = input + 4.0

            invoke_callback(
                callback_handler, AfterSuperCoolEvent(result, span_id)
            )

            return result
        end

        @test bar5(callback_handler, 9.0) == 13.0
        @test @allocated(bar5(callback_handler, 9.0)) === 0
    end
end

@testitem "invoke_callback should return the event" setup = [CallbacksTestUtils] begin
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
    @test invoke_callback(callbacks, CustomEvent(:other, 1)) ===
        CustomEvent(:other, 1)
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

    callbacks = (my_event = (event) -> begin
        event.state = sum(event.data)
    end,)

    event = MutableCustomEvent(:my_event, 3, 4; state = nothing)
    returned_event = invoke_callback(callbacks, event)

    @test returned_event === event
    @test event.state === 7
end

@testitem "Dict callback can mutate event state" setup = [CallbacksTestUtils] begin
    import ReactiveMP: invoke_callback

    callbacks = Dict{Symbol, Any}(
        :my_event => (event) -> begin
            event.state = prod(event.data)
        end
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
        sum_event = (event) -> nothing, prod_event = (event) -> nothing
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
        :sum_event => (event) -> nothing, :prod_event => (event) -> nothing
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
    ReactiveMP.handle_event(handler::MyCustomHandler, ::CustomEvent{:event2}) = push!(
        handler.events, :event2
    )

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

# -----------------------------------------------------------------------------
# `Base.show` golden-string tests for the Tier B events.
#
# Goal of these tests is to lock down the trace-logger output format so the
# RxInfer TBLogger extension can rely on `repr(ev)` to produce a single,
# greppable line per event. They use a fixed UUID for the span id so the
# truncated 4-char prefix is deterministic and the matching Before/After
# pair share it.
#
# Regression assertion: the output must NOT contain the raw struct dump
# pattern `MessageMapping{` or curly-brace type parameters from the event
# itself — that pattern is the telltale of the original `Base.show` default
# fallback that prompted issue #599 / RxInfer.jl#638.
# -----------------------------------------------------------------------------
@testmodule EventShowTestUtils begin
    using UUIDs
    import ReactiveMP:
        MessageMapping,
        Message,
        AnnotationDict,
        FormConstraintCheckEach,
        FormConstraintCheckLast,
        MessageProductContext

    struct MockVariable
        label::Symbol
    end

    # Fixed UUID so the truncated "ab12…" prefix is reproducible across runs.
    fixed_span() = UUID("ab123456-1234-5678-9abc-def012345678")

    # Build a trivial MessageMapping that exercises the new show method.
    # `F = Int` is a stand-in functional form: `show(::Type{Int})` yields "Int64".
    mapping(; vtag = :out, msgs = Val{(:μ, :τ)}(), marginals = nothing) = MessageMapping(
        Int, vtag, nothing, msgs, marginals, nothing, nothing, nothing, nothing,
        nothing,
    )

    msg(d) = Message(d, false, false, AnnotationDict())

    annotated_dict() = begin
        ann = AnnotationDict()
        ReactiveMP.annotate!(ann, :logscale, 1.0)
        ann
    end

    export MockVariable, fixed_span, mapping, msg, annotated_dict
end

@testitem "Base.show for Tier C supporting types" setup = [EventShowTestUtils] begin
    import ReactiveMP:
        MessageMapping,
        MessageProductContext,
        FormConstraintCheckEach,
        FormConstraintCheckLast,
        AnnotationDict

    # MessageMapping — single-line summary, names extracted from the Val{...} tuple.
    @test repr(EventShowTestUtils.mapping()) ==
        "MessageMapping(Int64, :out, msgs=[:μ, :τ])"
    @test repr(
        EventShowTestUtils.mapping(
            vtag = :μ, msgs = Val{(:out, :τ)}(), marginals = Val{(:q,)}()
        ),
    ) == "MessageMapping(Int64, :μ, msgs=[:out, :τ], marginals=[:q])"

    # MessageProductContext — surfaces only the strategy and fold direction.
    ctx = MessageProductContext()
    @test repr(ctx) == "MessageProductContext(strategy=CheckLast, fold=" *
        repr(ReactiveMP.MessagesProductFromLeftToRight()) *
        ")"

    # Form constraint check strategies — short, distinguishable labels.
    @test repr(FormConstraintCheckEach()) == "CheckEach"
    @test repr(FormConstraintCheckLast()) == "CheckLast"
end

@testitem "Base.show for Before/After event pairs" setup = [EventShowTestUtils] begin
    import ReactiveMP:
        BeforeMessageRuleCallEvent,
        AfterMessageRuleCallEvent,
        BeforeProductOfTwoMessagesEvent,
        AfterProductOfTwoMessagesEvent,
        BeforeProductOfMessagesEvent,
        AfterProductOfMessagesEvent,
        BeforeFormConstraintAppliedEvent,
        AfterFormConstraintAppliedEvent,
        BeforeMarginalComputationEvent,
        AfterMarginalComputationEvent,
        FormConstraintCheckEach,
        FormConstraintCheckLast,
        MessageProductContext,
        AnnotationDict

    span = EventShowTestUtils.fixed_span()
    var = EventShowTestUtils.MockVariable(:θ)
    ctx = MessageProductContext()
    mapping = EventShowTestUtils.mapping()
    left = EventShowTestUtils.msg(0.1)
    right = EventShowTestUtils.msg(0.2)
    result = EventShowTestUtils.msg(0.3)
    ann = AnnotationDict()

    # ---- MessageRuleCall ---------------------------------------------------
    before_rule = BeforeMessageRuleCallEvent(mapping, (left, right), nothing, span)
    after_rule = AfterMessageRuleCallEvent(
        mapping, (left, right), nothing, result, ann, span,
    )
    @test repr(before_rule) ==
        "BeforeMessageRuleCallEvent(mapping=MessageMapping(Int64, :out, msgs=[:μ, :τ]), nmsgs=2, nmarginals=0, span=ab12…)"
    @test repr(after_rule) ==
        "AfterMessageRuleCallEvent(mapping=MessageMapping(Int64, :out, msgs=[:μ, :τ]), nmsgs=2, nmarginals=0, result=Message(0.3), annotations=AnnotationDict(), span=ab12…)"

    # ---- ProductOfTwoMessages ---------------------------------------------
    before_p2 = BeforeProductOfTwoMessagesEvent(var, ctx, left, right, span)
    after_p2 = AfterProductOfTwoMessagesEvent(
        var, ctx, left, right, result, ann, span,
    )
    @test repr(before_p2) ==
        "BeforeProductOfTwoMessagesEvent(var=:θ, left=Message(0.1), right=Message(0.2), span=ab12…)"
    @test repr(after_p2) ==
        "AfterProductOfTwoMessagesEvent(var=:θ, left=Message(0.1), right=Message(0.2), result=Message(0.3), annotations=AnnotationDict(), span=ab12…)"

    # ---- ProductOfMessages -------------------------------------------------
    msgs = (left, right, result)
    before_pn = BeforeProductOfMessagesEvent(var, ctx, msgs, span)
    after_pn = AfterProductOfMessagesEvent(var, ctx, msgs, result, span)
    @test repr(before_pn) ==
        "BeforeProductOfMessagesEvent(var=:θ, nmessages=3, span=ab12…)"
    @test repr(after_pn) ==
        "AfterProductOfMessagesEvent(var=:θ, nmessages=3, result=Message(0.3), span=ab12…)"

    # ---- FormConstraintApplied --------------------------------------------
    before_form = BeforeFormConstraintAppliedEvent(
        var, ctx, FormConstraintCheckEach(), 0.5, span,
    )
    after_form = AfterFormConstraintAppliedEvent(
        var, ctx, FormConstraintCheckEach(), 0.5, 0.7, span,
    )
    @test repr(before_form) ==
        "BeforeFormConstraintAppliedEvent(var=:θ, strategy=CheckEach, dist=0.5, span=ab12…)"
    @test repr(after_form) ==
        "AfterFormConstraintAppliedEvent(var=:θ, strategy=CheckEach, dist=0.5, result=0.7, span=ab12…)"

    # ---- MarginalComputation ----------------------------------------------
    before_marg = BeforeMarginalComputationEvent(var, ctx, msgs, span)
    after_marg = AfterMarginalComputationEvent(var, ctx, msgs, result, span)
    @test repr(before_marg) ==
        "BeforeMarginalComputationEvent(var=:θ, nmessages=3, span=ab12…)"
    @test repr(after_marg) ==
        "AfterMarginalComputationEvent(var=:θ, nmessages=3, result=Message(0.3), span=ab12…)"

    # ---- Regression guard --------------------------------------------------
    # The original failure mode (issue #599 / RxInfer.jl#638) was that every
    # event landed in the trace logger as a raw struct dump containing
    # type-parameter braces. Make sure none of the new representations drift
    # back into that shape.
    for ev in (
        before_rule, after_rule, before_p2, after_p2, before_pn, after_pn,
        before_form, after_form, before_marg, after_marg,
    )
        rendered = repr(ev)
        @test !occursin("MessageMapping{", rendered)
        @test !occursin("Event{", rendered)
        @test count('\n', rendered) == 0
    end
end

@testitem "Base.show for nothing span_id remains total" begin
    import ReactiveMP:
        BeforeMessageRuleCallEvent, _show_span, _show_span_full, MessageMapping

    # If callbacks are disabled the span id is `nothing`. The compact and
    # full helpers must not throw — they degrade gracefully to a literal.
    short_io = IOBuffer()
    _show_span(short_io, nothing)
    @test String(take!(short_io)) == "span=nothing"

    full_io = IOBuffer()
    _show_span_full(full_io, nothing)
    @test String(take!(full_io)) == "span_id=nothing"
end
