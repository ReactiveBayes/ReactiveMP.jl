export AbstractMessage, Message, DeferredMessage
export getdata, is_clamped, is_initial, as_message

using Distributions
using Rocket

import Rocket: getrecent
import Base: ==, *, +, ndims, precision, length, size, show
import BayesBase: prod

# Text shared by the docstrings of `Message` and `Marginal` and of their accessors.
const DOC_CLAMPED = """
A clamped value is the result of computations on constants and observations alone: it never
changes during inference."""

const DOC_INITIAL = """
An initial value was set before inference, with [`ReactiveMP.set_initial_message!`](@ref) or
[`ReactiveMP.set_initial_marginal!`](@ref), or computed from clamped and initial values only,
at least one of them initial."""

const DOC_EQUALITY = """
`==` compares the data and the `is_clamped` and `is_initial` flags, not the annotations or the
log scale: those describe how a value was computed, not the belief it holds. Compare
[`getannotations`](@ref) and [`getlogscale`](@ref) explicitly where they matter."""

const DOC_STATISTICS = """
The statistics of the data are forwarded: `mean`, `median`, `mode`, `var`, `std`, `cov`,
`invcov`, `precision`, `entropy`, `params`, `mean_cov`, `mean_var`, `mean_invcov`,
`mean_precision`, `weightedmean` and its variants, `shape`, `scale`, `rate`, `probvec`,
`logdetcov`, `length`, `ndims`, `size`, `pdf` and `logpdf`."""

"""
    AbstractMessage

The supertype of the messages the engine passes: a [`Message`](@ref), which holds its value, and a
[`DeferredMessage`](@ref), which computes it on demand. [`as_message`](@ref) turns either into a
`Message`.
"""
abstract type AbstractMessage end

# The representation is a mutable struct with `const` fields: measured faster than an
# immutable one through the equality chain, and lighter everywhere
# (`scripts/benchmark_message_representation.jl`).
"""
    Message(data, is_clamped::Bool, is_initial::Bool)
    Message(data, is_clamped::Bool, is_initial::Bool, annotations::AnnotationDict)
    Message(data, is_clamped::Bool, is_initial::Bool, annotations::AnnotationDict, logscale)

A message along an edge of the factor graph: its data, usually a distribution, with the flags
and the metadata the engine tracks for it.

# Arguments

- `data`: the message itself, usually a distribution, and `missing` where it is not known,
  such as an unobserved data point;
- `is_clamped`: whether the message comes from constants and observations alone. $(DOC_CLAMPED)
- `is_initial`: whether the message was set before inference or computed from initial values
  only. $(DOC_INITIAL)
- `annotations`: the [`ReactiveMP.AnnotationDict`](@ref) of optional metadata, empty by default
  (see [`ReactiveMP.AbstractAnnotations`](@ref));
- `logscale`: the log of the constant the normalised `data` leaves out (see
  [`getlogscale`](@ref)): a number, an
  [`UndefinedLogScale`](@extref MessagePassingRulesBase.UndefinedLogScale) with its reason, or
  `nothing`, the default, where log scales are not tracked.

$(DOC_STATISTICS)

$(DOC_EQUALITY)

# Examples

```jldoctest
julia> message = Message(Gamma(10.0, 2.0), false, true)
Message(Distributions.Gamma{Float64}(α=10.0, θ=2.0))

julia> mean(message)
20.0

julia> is_clamped(message), is_initial(message)
(false, true)

julia> a = Message(NormalMeanVariance(0.0, 1.0), false, false);

julia> b = Message(NormalMeanVariance(0.0, 1.0), false, false, ReactiveMP.AnnotationDict(), -0.5);

julia> a == b, getlogscale(b)
(true, -0.5)
```

See also [`Marginal`](@ref), [`DeferredMessage`](@ref), [`as_message`](@ref).
"""
mutable struct Message{D, L} <: AbstractMessage
    const data::D
    const is_clamped::Bool
    const is_initial::Bool
    const annotations::AnnotationDict
    const logscale::L
end

Message(data, is_clamped::Bool, is_initial::Bool) =
    Message(data, is_clamped, is_initial, AnnotationDict(), nothing)
Message(data, is_clamped::Bool, is_initial::Bool, annotations::AnnotationDict) =
    Message(data, is_clamped, is_initial, annotations, nothing)

"""
    as_message(message::AbstractMessage) -> Message
    as_message(marginal::Marginal) -> Message

The [`Message`](@ref) a value stands for: a `Message` itself; the message a
[`DeferredMessage`](@ref) computes, computed on the first call and cached for the next; or a
`Marginal`'s data, flags, annotations and log scale as a message.
"""
function as_message end

as_message(message::Message) = message

"""
    getdata(message::Message)

The data of `message`, usually a distribution, or `missing`.
"""
getdata(message::Message) = message.data

"""
    is_clamped(message::Message) -> Bool

Whether `message` comes from constants and observations alone. $(DOC_CLAMPED)
"""
is_clamped(message::Message) = message.is_clamped

"""
    is_initial(message::Message) -> Bool

Whether `message` was set before inference or computed from initial values only. $(DOC_INITIAL)
"""
is_initial(message::Message) = message.is_initial

"""
    getannotations(message::Message) -> AnnotationDict

The [`ReactiveMP.AnnotationDict`](@ref) of `message`, empty unless annotation processors wrote to it.
"""
getannotations(message::Message) = message.annotations

"""
    getlogscale(message::Message)

The log scale of `message`: the log of the constant its normalised distribution leaves out, a
number, or an [`UndefinedLogScale`](@extref MessagePassingRulesBase.UndefinedLogScale) saying why
it is not known. Log scales are tracked when the node that computed the message was activated
with `logscales = true` (see [`ReactiveMP.FactorNodeActivationOptions`](@ref)).

# Throws

- `ArgumentError` when log scales are not tracked, the message carrying `nothing`.
"""
getlogscale(message::Message) = tracked_logscale(message.logscale)

typeofdata(message::Message) = typeof(getdata(message))

getdata(messages::NTuple{N, <:Message}) where {N} = map(getdata, messages)
# No inputs at all, which the `Message` and `Marginal` methods above would both match.
getdata(::Tuple{}) = ()
getdata(messages::AbstractArray{<:Message}) = map(getdata, messages)

function show(io::IO, message::Message)
    print(io, "Message(", getdata(message), ")")
    message.logscale === nothing || print(io, " with logscale = ", message.logscale)
    ann = getannotations(message)
    return if !isempty(ann)
        print(io, " with ", ann)
    end
end

# We need this dummy method as Julia is not smart enough to
# do that automatically if `data` is mutable
function Base.:(==)(left::Message, right::Message)
    return left.is_clamped == right.is_clamped &&
        left.is_initial == right.is_initial &&
        left.data == right.data
end

"""
    MessageProductContext(; prod_constraint = GenericProd(), form_constraint = UnspecifiedFormConstraint(), form_constraint_check_strategy = FormConstraintCheckLast(), fold_strategy = MessagesProductFromLeftToRight(), annotations = nothing, callbacks = nothing)

How a variable multiplies messages: the settings [`ReactiveMP.compute_product_of_messages`](@ref)
and [`ReactiveMP.compute_product_of_two_messages`](@ref) read. A random variable holds two, one for
its outbound messages and one for its marginal (see
[`RandomVariableActivationOptions`](@ref)).

# Keywords

- `prod_constraint`: the strategy `BayesBase.prod` multiplies two distributions with. Default
  `BayesBase.GenericProd()`, which keeps a product it has no closed form for as a
  `BayesBase.ProductOf`;
- `form_constraint`: the form the product is constrained to (see [`constrain_form`](@ref)).
  Default [`UnspecifiedFormConstraint`](@ref), which leaves it as it is;
- `form_constraint_check_strategy`: when the form constraint applies, once to the whole product,
  [`FormConstraintCheckLast`](@ref), the default, or after each pairwise product,
  [`FormConstraintCheckEach`](@ref);
- `fold_strategy`: the order the messages are multiplied in:
  [`ReactiveMP.MessagesProductFromLeftToRight`](@ref), the default,
  [`ReactiveMP.MessagesProductFromRightToLeft`](@ref), or a function `f(variable, context,
  messages)` that multiplies them with [`ReactiveMP.compute_product_of_two_messages`](@ref) in any
  order;
- `annotations`: the annotation processors that merge the two sides' annotations in each
  pairwise product, a collection of [`ReactiveMP.AbstractAnnotations`](@ref). Default `nothing`,
  which gives each product empty annotations, except that a product with a `missing` side keeps
  the other side's;
- `callbacks`: the handler of the product events, such as
  [`ReactiveMP.BeforeProductOfTwoMessagesEvent`](@ref) (see [`ReactiveMP.invoke_callback`](@ref)).
  Default `nothing`, none.

RxInfer builds it from a variable's form constraint, with [`default_prod_constraint`](@ref) and
[`default_form_check_strategy`](@ref) of that constraint.

See also [`ReactiveMP.compute_product_of_messages`](@ref),
[`ReactiveMP.compute_product_of_two_messages`](@ref).
"""
Base.@kwdef struct MessageProductContext{C, F, S, L, N, A}
    prod_constraint::C = BayesBase.GenericProd()
    form_constraint::F = UnspecifiedFormConstraint()
    form_constraint_check_strategy::S = FormConstraintCheckLast()
    fold_strategy::L = MessagesProductFromLeftToRight()
    annotations::N = nothing
    callbacks::A = nothing
end

function Base.show(io::IO, ctx::MessageProductContext)
    print(io, "MessageProductContext(strategy=")
    show(io, ctx.form_constraint_check_strategy)
    print(io, ", fold=")
    show(io, ctx.fold_strategy)
    if !get(io, :compact, false)
        print(io, ", form_constraint=")
        show(io, ctx.form_constraint)
        print(io, ", prod_constraint=")
        show(io, ctx.prod_constraint)
    end
    print(io, ")")
    return nothing
end

"""
    compute_product_of_two_messages(variable::AbstractVariable, context::MessageProductContext, left, right) -> Message

Multiply two messages for `variable`, with `BayesBase.prod` under `context.prod_constraint`. A
[`DeferredMessage`](@ref) is computed first (see [`as_message`](@ref)). The product is a
[`Message`](@ref), not necessarily normalised; `context.form_constraint` applies to it here when
the strategy is [`FormConstraintCheckEach`](@ref).

The product is clamped when both messages are, and initial when it is not clamped and each
side is clamped or initial.

Its log scale is the sum of the two messages' log scales and the product's own,
`BayesBase.compute_logscale(new, left, right)`. A `missing` side leaves the other side's; either
side's `nothing`, log scales not tracked, gives `nothing`. It is undefined, an
[`UndefinedLogScale`](@extref MessagePassingRulesBase.UndefinedLogScale), when either side's is,
when the pair has no `compute_logscale`, and when a form constraint changes the product, returning
something other than it was given.

Its annotations are merged by the processors in `context.annotations` (see
[`ReactiveMP.post_product_annotations!`](@ref)).

`variable` names the variable the product is for; it reaches the callbacks, which receive
[`ReactiveMP.BeforeProductOfTwoMessagesEvent`](@ref) and
[`ReactiveMP.AfterProductOfTwoMessagesEvent`](@ref), and, with the strategy
[`FormConstraintCheckEach`](@ref), the form constraint events.

See also [`ReactiveMP.MessageProductContext`](@ref), [`ReactiveMP.compute_product_of_messages`](@ref).
"""
function compute_product_of_two_messages(
        variable::AbstractVariable,
        context::MessageProductContext,
        left::Message,
        right::Message,
    )
    span_id = generate_span_id(context.callbacks)
    invoke_callback(
        context.callbacks,
        BeforeProductOfTwoMessagesEvent(
            variable, context, left, right, span_id
        ),
    )

    # We propagate clamped message, in case if both are clamped
    is_prod_clamped = is_clamped(left) && is_clamped(right)
    # We propagate initial message, in case if both are initial or left is initial and right is clameped or vice-versa
    is_prod_initial =
        !is_prod_clamped &&
        (is_clamped_or_initial(left)) &&
        (is_clamped_or_initial(right))

    # process distributions
    left_dist = getdata(left)
    right_dist = getdata(right)
    new_dist = prod(context.prod_constraint, left_dist, right_dist)
    new_logscale = product_logscale(new_dist, left_dist, right_dist, left.logscale, right.logscale)

    if context.form_constraint_check_strategy === FormConstraintCheckEach()
        form_span_id = generate_span_id(context.callbacks)
        invoke_callback(
            context.callbacks,
            BeforeFormConstraintAppliedEvent(
                variable,
                context,
                FormConstraintCheckEach(),
                new_dist,
                form_span_id,
            ),
        )
        unconstrained_dist = new_dist
        new_dist = constrain_form(context.form_constraint, new_dist)
        new_logscale = constrained_logscale(unconstrained_dist, new_dist, new_logscale)
        invoke_callback(
            context.callbacks,
            AfterFormConstraintAppliedEvent(
                variable,
                context,
                FormConstraintCheckEach(),
                unconstrained_dist,
                new_dist,
                form_span_id,
            ),
        )
    end

    # process annotations
    left_ann = getannotations(left)
    right_ann = getannotations(right)
    new_ann = post_product_annotations!(context.annotations, left_ann, right_ann, new_dist, left_dist, right_dist)
    result = Message(new_dist, is_prod_clamped, is_prod_initial, new_ann, new_logscale)

    invoke_callback(
        context.callbacks,
        AfterProductOfTwoMessagesEvent(
            variable, context, left, right, result, new_ann, span_id
        ),
    )

    return result
end

# Sometimes we call the product on the `DeferredMessage` that need to be casted to a `Message`
function compute_product_of_two_messages(
        variable::AbstractVariable, context::MessageProductContext, left, right
    )
    return compute_product_of_two_messages(
        variable, context, as_message(left), as_message(right)
    )
end

"""
    compute_product_of_messages(variable::AbstractVariable, context::MessageProductContext, messages) -> Message

Multiply a collection of messages for `variable`, pairwise with
[`ReactiveMP.compute_product_of_two_messages`](@ref), in the order `context.fold_strategy` gives.
With the strategy [`FormConstraintCheckLast`](@ref), `context.form_constraint` applies once, to the
whole product, and a form constraint that changes it leaves its log scale undefined.

The callbacks receive [`ReactiveMP.BeforeProductOfMessagesEvent`](@ref) and
[`ReactiveMP.AfterProductOfMessagesEvent`](@ref) around the whole product.

# Examples

```jldoctest
julia> x = randomvar();

julia> messages = (Message(NormalMeanVariance(0.0, 1.0), false, false), Message(NormalMeanVariance(2.0, 1.0), false, false));

julia> product = ReactiveMP.compute_product_of_messages(x, ReactiveMP.MessageProductContext(), messages);

julia> mean(product) ≈ 1.0 && var(product) ≈ 0.5
true
```

See also [`ReactiveMP.MessageProductContext`](@ref), [`ReactiveMP.MessagesProductFromLeftToRight`](@ref).
"""
function compute_product_of_messages(
        variable::AbstractVariable, context::MessageProductContext, messages
    )
    span_id = generate_span_id(context.callbacks)
    invoke_callback(
        context.callbacks,
        BeforeProductOfMessagesEvent(variable, context, messages, span_id),
    )

    result = as_message(
        compute_product_of_messages(
            context.fold_strategy, variable, context, messages
        ),
    )

    if context.form_constraint_check_strategy === FormConstraintCheckLast()
        dist = getdata(result)
        form_span_id = generate_span_id(context.callbacks)
        invoke_callback(
            context.callbacks,
            BeforeFormConstraintAppliedEvent(
                variable, context, FormConstraintCheckLast(), dist, form_span_id
            ),
        )
        constrained_dist = constrain_form(context.form_constraint, dist)
        invoke_callback(
            context.callbacks,
            AfterFormConstraintAppliedEvent(
                variable,
                context,
                FormConstraintCheckLast(),
                dist,
                constrained_dist,
                form_span_id,
            ),
        )
        result = Message(
            constrained_dist,
            is_clamped(result),
            is_initial(result),
            getannotations(result),
            constrained_logscale(dist, constrained_dist, result.logscale),
        )
    end

    invoke_callback(
        context.callbacks,
        AfterProductOfMessagesEvent(
            variable, context, messages, result, span_id
        ),
    )

    return result
end

"""
    MessagesProductFromLeftToRight()

The fold strategy that multiplies messages from the first to the last, with `foldl`: the
default of [`ReactiveMP.MessageProductContext`](@ref).

See also [`ReactiveMP.MessagesProductFromRightToLeft`](@ref).
"""
struct MessagesProductFromLeftToRight end

function compute_product_of_messages(
        ::MessagesProductFromLeftToRight,
        variable::AbstractVariable,
        context::MessageProductContext,
        messages,
    )
    return foldl(
        (left, right) ->
        compute_product_of_two_messages(variable, context, left, right),
        messages,
    )
end

"""
    MessagesProductFromRightToLeft()

The fold strategy that multiplies messages from the last to the first, with `foldr`.

See also [`ReactiveMP.MessagesProductFromLeftToRight`](@ref), [`ReactiveMP.MessageProductContext`](@ref).
"""
struct MessagesProductFromRightToLeft end

function compute_product_of_messages(
        ::MessagesProductFromRightToLeft,
        variable::AbstractVariable,
        context::MessageProductContext,
        messages,
    )
    return foldr(
        (left, right) ->
        compute_product_of_two_messages(variable, context, left, right),
        messages,
    )
end

"""
    compute_product_of_messages(f::Function, variable::AbstractVariable, context::MessageProductContext, messages)

The product under a custom fold strategy, `context.fold_strategy = f`: calls
`f(variable, context, messages)`, which multiplies the messages with
[`ReactiveMP.compute_product_of_two_messages`](@ref) in whatever order it chooses and returns the
product.
"""
function compute_product_of_messages(
        f::Function,
        variable::AbstractVariable,
        context::MessageProductContext,
        messages,
    )
    return f(variable, context, messages)
end

Distributions.pdf(message::Message, x) = Distributions.pdf(getdata(message), x)
Distributions.logpdf(message::Message, x) = Distributions.logpdf(getdata(message), x)

MacroHelpers.@proxy_methods Message getdata [
    BayesBase.mean,
    BayesBase.median,
    BayesBase.mode,
    BayesBase.shape,
    BayesBase.scale,
    BayesBase.rate,
    BayesBase.var,
    BayesBase.std,
    BayesBase.cov,
    BayesBase.invcov,
    BayesBase.logdetcov,
    BayesBase.entropy,
    BayesBase.params,
    BayesBase.mean_cov,
    BayesBase.mean_var,
    BayesBase.mean_invcov,
    BayesBase.mean_precision,
    BayesBase.weightedmean_cov,
    BayesBase.weightedmean_var,
    BayesBase.weightedmean_invcov,
    BayesBase.weightedmean_precision,
    BayesBase.probvec,
    BayesBase.weightedmean,
    Base.precision,
    Base.length,
    Base.ndims,
    Base.size,
]

# Eltype is special here, because it should be only defined on types
# Otherwise it causes invalidations and slower compile times
Base.eltype(::Type{<:Message{D}}) where {D} = Base.eltype(D)

Distributions.mean(fn::Function, message::Message) = mean(fn, getdata(message))

## Deferred Message

"""
    DeferredMessage(messages, marginals, mapping)

A message computed on demand: what a factor node emits on an outbound stream, so that a message
nobody reads is never computed. [`as_message`](@ref) computes it on the first call, as
`mapping(getrecent(messages), getrecent(marginals))`, from the latest values of its inputs at
that time, and caches the resulting [`Message`](@ref) for the next calls.

# Arguments

- `messages`: the stream of the rule's inbound messages, or `nothing` for none;
- `marginals`: the stream of the rule's marginals, or `nothing` for none;
- `mapping`: a function of the two, usually a [`ReactiveMP.MessageMapping`](@ref), that returns
  the `Message`.

A variable computes the deferred messages it receives when it multiplies them.
"""
mutable struct DeferredMessage{R, S, F} <: AbstractMessage
    const messages::R
    const marginals::S
    const mappingFn::F
    cache::Union{Nothing, Message}
end

DeferredMessage(messages::R, marginals::S, mappingFn::F) where {R, S, F} =
    DeferredMessage(messages, marginals, mappingFn, nothing)

function Base.show(io::IO, message::DeferredMessage)
    cache = getcache(message)
    return if isnothing(cache)
        print(
            io, "DeferredMessage([ use `as_message` to compute the message ])"
        )
    else
        print(io, "DeferredMessage(", getdata(cache), ")")
    end
end

getcache(message::DeferredMessage) = message.cache
setcache!(message::DeferredMessage, cache::Message) = message.cache = cache

function as_message(message::DeferredMessage)::Message
    return as_message(message, getcache(message))
end

function as_message(message::DeferredMessage, cache::Message)::Message
    return cache
end

function as_message(message::DeferredMessage, cache::Nothing)::Message
    return as_message(
        message,
        cache,
        getrecent(message.messages),
        getrecent(message.marginals),
    )
end

function as_message(
        message::DeferredMessage, cache::Nothing, messages, marginals
    )::Message
    computed = message.mappingFn(messages, marginals)
    setcache!(message, computed)
    return computed
end


## Message observable

"""
    MessageObservable(M::Type = AbstractMessage)

The stream of the messages along one connection between a variable and a factor node, with
values of type `M`. Every subscriber shares one upstream subscription, and the latest message is
kept, so `Rocket.getrecent` returns it and a late subscriber receives it at once.

The stream is lazy: activation connects it to its source, the node's rule or the variable's
product. Before that, [`ReactiveMP.set_initial_message!`](@ref) can seed it with an initial
message, which is what a rule reads before any message has been computed.

A random and a data variable allocate one per connection
([`ReactiveMP.create_new_stream_of_inbound_messages!`](@ref)); a constant has one, shared by all
its connections.

See also [`ReactiveMP.MarginalObservable`](@ref).
"""
struct MessageObservable{M <: AbstractMessage} <: Subscribable{M}
    subject::Rocket.RecentSubjectInstance{M, Subject{M, AsapScheduler, AsapScheduler}}
    stream::LazyObservable{M}
end

MessageObservable(::Type{M} = AbstractMessage) where {M} =
    MessageObservable{M}(RecentSubject(M), lazy(M))

Rocket.getrecent(observable::MessageObservable) =
    Rocket.getrecent(observable.subject)

@inline Rocket.on_subscribe!(observable::MessageObservable, actor) =
    subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.Actor{<:AbstractMessage}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.NextActor{<:AbstractMessage}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.ErrorActor{<:AbstractMessage}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.CompletionActor{<:AbstractMessage}) = Rocket.on_subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.Subject{<:AbstractMessage}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.BehaviorSubjectInstance{<:AbstractMessage}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.PendingSubjectInstance{<:AbstractMessage}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.RecentSubjectInstance{<:AbstractMessage}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.ReplaySubjectInstance{<:AbstractMessage}) = Rocket.on_subscribe!(observable.stream, actor)

function connect!(message::MessageObservable, source)
    set!(message.stream, source |> multicast(message.subject) |> ref_count())
    return nothing
end

function set_initial_message!(message::MessageObservable, value)
    next!(message.subject, Message(value, false, true, AnnotationDict(), INITIAL_LOGSCALE))
    return nothing
end

## Message Mapping structure
## A callable structure rather than a closure, which `activate!` (src/nodes/nodes.jl) would
## capture with types Julia cannot fully infer (https://github.com/JuliaLang/julia/issues/42559).
"""
    MessageMapping

What computes a node's message towards one interface, called with the latest inbound messages
and marginals the rule depends on: `mapping(messages, marginals) -> Message`. A factor node
builds one per interface at activation and wraps it in each [`DeferredMessage`](@ref) it emits.

It holds the node type, the rule's target, the names of its inputs, the algorithm, the
annotation processors, the factor node, the callbacks, the [`ReactiveMP.EngineDiagnostics`](@ref),
the rule context (see [`ReactiveMP.node_context`](@ref)), the rule fallback, the rule's scratch
(reused from call to call) and whether log scales are tracked.

A call:

1. returns a `missing` message, and runs no rule, when an input is `missing`;
2. resolves the rule with
   [`find_message_rule`](@extref MessagePassingRulesBase.find_message_rule) under the node's
   algorithm, from the inputs' types;
3. where no rule matches, gives the rule fallback's message, if one is set, with an undefined log
   scale; otherwise throws a
   [`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError) listing the near
   misses;
4. checks the rule: against the diagnostics, for a declared reading of log scales that are not
   tracked, and for a declared context service the node's context does not supply
   ([`check_services`](@extref MessagePassingRulesBase.check_services));
5. runs it with [`execute_rule`](@extref MessagePassingRulesBase.execute_rule), with the
   annotation processors before and after it.

The message is clamped when every input is, and initial when it is not clamped and every input
is clamped or initial. When log scales are tracked, its log scale is the one the rule declares.

The callbacks receive [`ReactiveMP.BeforeMessageRuleCallEvent`](@ref) and
[`ReactiveMP.AfterMessageRuleCallEvent`](@ref) around each call, the latter also for a `missing`
message.

See also [`Message`](@ref), [`ReactiveMP.rule_arguments`](@ref).
"""
struct MessageMapping{F, T, N, M, A, X, R, E, G, B, S}
    target::T
    msgs_names::N
    marginals_names::M
    algorithm::A
    annotations::X
    factornode::R
    callbacks::E
    diagnostics::EngineDiagnostics
    context::G
    rulefallback::B
    scratch::ScratchSlot
    logscales::S
end

message_mapping_fform(::MessageMapping{F}) where {F} = F
message_mapping_fform(::MessageMapping{F}) where {F <: Function} = F.instance

function Base.show(io::IO, mapping::MessageMapping)
    print(io, "MessageMapping(")
    print(io, message_mapping_fform(mapping))
    print(io, ", ", repr(mapping.target))
    if mapping.msgs_names !== nothing
        print(io, ", msgs=", collect(unval(mapping.msgs_names)))
    end
    if mapping.marginals_names !== nothing
        print(io, ", marginals=", collect(unval(mapping.marginals_names)))
    end
    if !get(io, :compact, false) && mapping.algorithm !== nothing
        print(io, ", algorithm=", mapping.algorithm)
    end
    print(io, ")")
    return nothing
end

# `context` holds the services the rules run with, merged over the engine's (`node_context`);
# `rulefallback` gives the message where no rule matches, `nothing` for none; `logscales` tracks
# log scales.
MessageMapping(::Type{F}, target::T, msgs_names::N, marginals_names::M, algorithm::A, annotations::X, factornode::R, callbacks::E, diagnostics::EngineDiagnostics = EngineDiagnostics(), context = nothing, rulefallback::B = nothing, logscales::Bool = false) where {F, T, N, M, A, X, R, E, B} =
    (c = node_context(factornode, context); s = Val(logscales); MessageMapping{F, T, N, M, A, X, R, E, typeof(c), B, typeof(s)}(target, msgs_names, marginals_names, algorithm, annotations, factornode, callbacks, diagnostics, c, rulefallback, ScratchSlot(), s))

MessageMapping(::F, target::T, msgs_names::N, marginals_names::M, algorithm::A, annotations::X, factornode::R, callbacks::E, diagnostics::EngineDiagnostics = EngineDiagnostics(), context = nothing, rulefallback::B = nothing, logscales::Bool = false) where {F <: Function, T, N, M, A, X, R, E, B} =
    (c = node_context(factornode, context); s = Val(logscales); MessageMapping{F, T, N, M, A, X, R, E, typeof(c), B, typeof(s)}(target, msgs_names, marginals_names, algorithm, annotations, factornode, callbacks, diagnostics, c, rulefallback, ScratchSlot(), s))

tracks_logscales(mapping::MessageMapping) = mapping.logscales isa Val{true}

# The fallback's message where no rule matched; the not-found error where it has none either.
function fallback_message(fallback, notfound, fform, target, args)
    message = fallback(fform, target, args)
    isnothing(message) && throw(RuleNotFoundError(notfound))
    return message
end

function (mapping::MessageMapping)(messages, marginals)
    # Message is clamped if all of the inputs are clamped
    is_message_clamped =
        __check_all(is_clamped, messages) && __check_all(is_clamped, marginals)

    # Message is initial if it is not clamped and all of the inputs are either clamped or initial
    is_message_initial =
        !is_message_clamped && (
        __check_all(is_clamped_or_initial, messages) &&
            __check_all(is_clamped_or_initial, marginals)
    )

    span_id = generate_span_id(mapping.callbacks)
    invoke_callback(
        mapping.callbacks,
        BeforeMessageRuleCallEvent(mapping, messages, marginals, span_id),
    )

    annotations = AnnotationDict()

    # Run annotation processors before the rule has been executed
    if !isnothing(mapping.annotations)
        for p in mapping.annotations
            pre_rule_annotations!(p, annotations, mapping, messages, marginals)
        end
    end

    result, logscale = if has_missing_inputs(messages) || has_missing_inputs(marginals)
        missing, nothing
    else
        fform = message_mapping_fform(mapping)
        args = rule_arguments(mapping.msgs_names, messages, mapping.marginals_names, marginals, mapping.logscales)
        found = MessagePassingRulesBase.find_message_rule(fform, mapping.target, mapping.algorithm, args)
        if found isa MessagePassingRulesBase.RuleNotFound && !isnothing(mapping.rulefallback)
            fallback_message(mapping.rulefallback, found, fform, mapping.target, args), (tracks_logscales(mapping) ? FALLBACK_LOGSCALE : nothing)
        else
            spec = audit_rule(mapping.diagnostics, resolve_rule(found))
            MessagePassingRulesBase.check_reads_logscale(spec, args)
            ctx = mapping.context
            MessagePassingRulesBase.check_services(spec, ctx)
            ann = rule_annotations(mapping.msgs_names, messages, mapping.marginals_names, marginals, annotations)
            algorithm = MessagePassingRulesBase.rule_algorithm(spec, mapping.algorithm)
            scratch = scratch_for!(mapping.scratch, spec, algorithm, ctx, args, mapping.target, mapping.diagnostics.checked_buffers)
            if tracks_logscales(mapping)
                MessagePassingRulesBase.execute_rule_with_logscale(spec, nothing, scratch, algorithm, ctx, args, ann, mapping.target)
            else
                MessagePassingRulesBase.execute_rule(spec, nothing, scratch, algorithm, ctx, args, ann, mapping.target), nothing
            end
        end
    end

    # Run annotation processors after the rule has been executed. Skip them entirely when the
    # rule was short-circuited to `missing`: no rule ran, so there is nothing to annotate.
    if !isnothing(mapping.annotations) && result !== missing
        for p in mapping.annotations
            post_rule_annotations!(
                p, annotations, mapping, messages, marginals, result
            )
        end
    end

    invoke_callback(
        mapping.callbacks,
        AfterMessageRuleCallEvent(
            mapping, messages, marginals, result, annotations, logscale, span_id
        ),
    )

    return Message(result, is_message_clamped, is_message_initial, annotations, logscale)
end
