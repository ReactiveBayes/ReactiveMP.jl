"""
    ReactiveMP

A reactive message passing engine for Bayesian inference on factor graphs. A graph is built from
variables ([`randomvar`](@ref), [`datavar`](@ref), [`constvar`](@ref)) and factor nodes
([`factornode`](@ref)), and activated: messages and marginals are then streams, and each new
observation ([`new_observation!`](@ref)) propagates through them. A factor node computes each
outbound message with the update rule
[`MessagePassingRulesBase`](https://reactivebayes.github.io/MessagePassingRulesBase.jl/dev/) finds for it,
from the latest messages and marginals it depends on; a variable multiplies its inbound messages
into its marginal; [`bethe_free_energy`](@ref) is the stream of the variational objective.

The engine defines no node and no rule: they come from the rule packages, such as
[`StandardMessagePassingRules`](https://reactivebayes.github.io/StandardMessagePassingRules.jl/dev/),
which depend on the base package only. Most models are written with RxInfer, which builds and
runs the graph on this engine.

# Examples

A normal prior on `x` and one observation `y ~ N(x, 1)`:

```jldoctest; setup = :(using StandardMessagePassingRules, Rocket)
julia> import ReactiveMP: activate!, FactorNodeActivationOptions, get_stream_of_marginals

julia> x, y = randomvar(), datavar();

julia> prior = factornode(NormalMeanVariance, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))]);

julia> likelihood = factornode(NormalMeanVariance, [(:out, y), (:μ, x), (:v, constvar(1.0))]);

julia> activate!(x, RandomVariableActivationOptions()); activate!(y, DataVariableActivationOptions());

julia> foreach(node -> activate!(node, FactorNodeActivationOptions()), (prior, likelihood));

julia> posterior = Ref{Any}(); subscribe!(get_stream_of_marginals(x), (q) -> posterior[] = q);

julia> new_observation!(y, 2.0)

julia> mean(posterior[]) ≈ 20 / 11 && var(posterior[]) ≈ 10 / 11
true
```
"""
module ReactiveMP

# List global dependencies here
using TinyHugeNumbers, LinearAlgebra
using BayesBase
using UUIDs
using Compat: @compat

import MessagePassingRulesBase


# Reexport `tiny` and `huge` from the `TinyHugeNumbers`
export tiny, huge

# The engine's interface that RxInfer and extensions build on, which the documentation tells
# them to use: public, not exported. `@compat` makes the declaration parse on Julia 1.10.
@compat public activate!, FactorNodeActivationOptions, MessageProductContext, EngineDiagnostics,
    node_context, set_initial_marginal!, set_initial_message!, get_stream_of_marginals,
    get_stream_of_predictions, Event, event_name, invoke_callback, handle_event,
    merge_callbacks, generate_span_id, AnnotationDict, annotate!, get_annotation,
    has_annotation, AbstractAnnotations, pre_rule_annotations!, post_rule_annotations!,
    post_product_annotations!, AbstractVariable, degree, israndom, isdata, isconst,
    preprocess_form_constraints, WrappedFormConstraint, prepare_context, MessageMapping,
    MarginalMapping, rule_arguments, compute_product_of_messages,
    compute_product_of_two_messages, MessageObservable, MarginalObservable,
    get_stream_of_inbound_messages, get_stream_of_outbound_messages, getvariable, getinterface,
    name, getlocalclusters, AbstractStreamPostprocessor,
    postprocess_stream_of_outbound_messages, postprocess_stream_of_marginals,
    postprocess_stream_of_scores, ScheduleOnStreamPostprocessor,
    AfterFormConstraintAppliedEvent, AfterMarginalComputationEvent, AfterMessageRuleCallEvent,
    AfterProductOfMessagesEvent, AfterProductOfTwoMessagesEvent,
    BeforeFormConstraintAppliedEvent, BeforeMarginalComputationEvent,
    BeforeMessageRuleCallEvent, BeforeProductOfMessagesEvent, BeforeProductOfTwoMessagesEvent

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
