"""
    ReactiveMP

A reactive message passing engine for Bayesian inference on factor graphs. Messages and
marginals are streams: a factor node computes each outbound message with the update rule that
`MessagePassingRulesBase` finds for it, from the latest messages and marginals it depends on,
and a variable combines its inbound messages into its marginal. Rules come from the rule
packages, which depend on the base package only, never on the engine.
"""
module ReactiveMP

# List global dependencies here
using TinyHugeNumbers, LinearAlgebra
using BayesBase
using UUIDs

import MessagePassingRulesBase


# Reexport `tiny` and `huge` from the `TinyHugeNumbers`
export tiny, huge

include("fixes.jl")
include("helpers/macrohelpers.jl")
include("helpers/helpers.jl")

include("constraints/form.jl")

include("callbacks.jl")
include("postprocessors.jl")
include("variable.jl")
include("annotations.jl")
include("logscale.jl")
include("annotations/input_arguments.jl")
include("scratch.jl")
include("diagnostics.jl")
include("message.jl")
include("marginal.jl")

# A marginal is formed in its public type (`MessagePassingRulesBase.public_equivalent`): an
# efficient working type such as `WishartFast` does not leave the product of messages.
as_marginal(message::Message) = Marginal(MessagePassingRulesBase.public_equivalent(getdata(message)), is_clamped(message), is_initial(message), getannotations(message), message.logscale)
as_message(marginal::Marginal) = Message(getdata(marginal), is_clamped(marginal), is_initial(marginal), getannotations(marginal), marginal.logscale)

getdata(::Nothing) = nothing
getdata(collection::Tuple) = map(getdata, collection)
getdata(collection::AbstractArray) = map(getdata, collection)

# TupleTools.prod is a more efficient version of Base.all for Tuple here
is_clamped(tuple::Tuple) = TupleTools.prod(map(is_clamped, tuple))
is_initial(tuple::Tuple) = TupleTools.prod(map(is_initial, tuple))

include("rule_arguments.jl")
include("context.jl")

# Predefined postprocessors
include("postprocessors/scheduled.jl")

# Equality node is a special case and needs to be included before random variable implementation
include("nodes/equality.jl")

include("variables/random.jl")
include("variables/constant.jl")
include("variables/data.jl")

include("nodes/nodes.jl")

include("score/score.jl")
include("score/variable.jl")
include("score/node.jl")
include("score/bethe.jl")

function __init__()
    Base.Experimental.register_error_hint(
        MethodError
    ) do io, exc, argtypes, kwargs
        if exc.f === ReactiveMP.handle_event && length(argtypes) >= 2
            event_type = argtypes[2]
            event_hint = if event_type <: ReactiveMP.Event
                "Event{$(repr(ReactiveMP.event_name(event_type)))}"
            else
                string(event_type)
            end
            errmsg = """

            `ReactiveMP.handle_event` was called with a callback handler of type `$(argtypes[1])` for event `$(event_type)`, but no matching method was found. This can happen if:

            1. You implemented a custom callback handler but forgot to define `handle_event` for this specific event type.
               Make sure your handler has a method like:
                 ReactiveMP.handle_event(::$(argtypes[1]), event::$(event_hint)) = ...

            2. You meant to pass a `NamedTuple` as the callbacks handler but forgot the trailing comma.
               In Julia, `(key = value)` is parsed as a plain assignment, not a NamedTuple.
               Use `(key = value,)` (with a trailing comma) instead.
            """
            println(io, errmsg)
        end
    end
end

end
