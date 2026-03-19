export AbstractMessage, Message, DeferredMessage
export getdata, is_clamped, is_initial, as_message

using Distributions
using Rocket

import Rocket: getrecent
import Base: ==, *, +, ndims, precision, length, size, show

"""
An abstract supertype for all concrete message types.
"""
abstract type AbstractMessage end

"""
    Message(data, is_clamped, is_initial, addons)

An implementation of a message in variational message passing framework.

# Arguments
- `data::D`: message always holds some data object associated with it, which is usually a probability distribution, but can also be an arbitrary function
- `is_clamped::Bool`, specifies if this message was the result of constant computations (e.g. clamped constants)
- `is_initial::Bool`, specifies if this message was used for initialization
- `addons::A`, specifies the addons of the message, which may carry extra bits of information, e.g. debug information, memory, etc.

# Example 

```jldoctest
julia> distribution = Gamma(10.0, 2.0)
Distributions.Gamma{Float64}(α=10.0, θ=2.0)

julia> message = Message(distribution, false, true, nothing)
Message(Distributions.Gamma{Float64}(α=10.0, θ=2.0))

julia> mean(message) 
20.0

julia> getdata(message)
Distributions.Gamma{Float64}(α=10.0, θ=2.0)

julia> is_clamped(message)
false

julia> is_initial(message)
true

```
"""
mutable struct Message{D, A} <: AbstractMessage # `mutable` structure here appears to be more performance 
    const data       :: D                       # in `RxInfer` benchmarks
    const is_clamped :: Bool                    # could be revised at some point though
    const is_initial :: Bool
    const addons     :: A
end

"""
    as_message(::AbstractMessage)

A function that converts an abstract message to an instance of `Message`.
"""
function as_message end

as_message(message::Message) = message

"""
    getdata(message::Message)    

Returns `data` associated with the `message`.
"""
getdata(message::Message) = message.data

"""
    is_clamped(message::Message)

Checks if `message` is clamped or not.
"""
is_clamped(message::Message) = message.is_clamped

"""
    is_initial(message::Message)

Checks if `message` is initial or not.
"""
is_initial(message::Message) = message.is_initial

"""
    getaddons(message::Message)

Returns `addons` associated with the `message`.
"""
getaddons(message::Message) = message.addons

typeofdata(message::Message) = typeof(getdata(message))

getdata(messages::NTuple{N, <:Message}) where {N} = map(getdata, messages)
getdata(messages::AbstractArray{<:Message})       = map(getdata, messages)

# Base.show(io::IO, message::Message) = print(io, string("Message(", getdata(message), ") with ", string(getaddons(message))))
function show(io::IO, message::Message)
    print(io, string("Message(", getdata(message), ")"))
    if !isnothing(getaddons(message))
        print(io, ") with ", string(getaddons(message)))
    end
end

# We need this dummy method as Julia is not smart enough to 
# do that automatically if `data` is mutable
function Base.:(==)(left::Message, right::Message)
    return left.is_clamped == right.is_clamped &&
           left.is_initial == right.is_initial &&
           left.data == right.data &&
           left.addons == right.addons
end

"""
    MessageProductContext(kwargs...)

The structure that defines the context for the product of **two** messages within ReactiveMP.
The product is executed with the [`ReactiveMP.compute_product_of_messages`](@ref) function and 
uses the `BayesBase.prod` under the hood. See BayesBase product API documentation for detailed description.

The following `kwargs` are supported:
- `prod_constraint`: defines the first argument for the `BayesBase.prod` function (default is `BayesBase.GenericProd`)
- `form_constraint`: defines the form constraint to be applied on the result of computation, default is [`ReactiveMP.UnspecifiedFormConstraint`](@ref)
- `form_constraint_check_strategy`: defines the strategy to check the specified form constraint, either [`ReactiveMP.FormConstraintCheckLast`](@ref) or [`ReactiveMP.FormConstraintCheckEach`](@ref), default is [`ReactiveMP.FormConstraintCheckLast`](@ref)
    + [`ReactiveMP.FormConstraintCheckLast`](@ref) will only call [`ReactiveMP.constrain_form`](@ref) at the end of the `[ReactiveMP.compute_product_of_messages]`
    + [`ReactiveMP.FormConstraintCheckEach`](@ref) will call [`ReactiveMP.constrain_form`](@ref) at each of the [`ReactiveMP.compute_product_of_two_messages`](@ref)
- `fold_strategy`: defines the strategy (or simply speaking the direction) of the messages product for [`ReactiveMP.compute_product_of_messages`](@ref), default is [`MessagesProductFromLeftToRight`](@ref). Can be a custom function that accepts a `variable`, `context` and collection of `messages` and does arbitrary order, but still needs to call the [`ReactiveMP.compute_product_of_two_messages`](@ref) under the hood (unless you do some experimental stuff). By the way it is called __fold__ to reflect the computer science term with "left-fold" or "right-fold" (and we use the builtin Julia `foldl` and `foldr` functions for that).
- `callbacks`: callbacks handler, see [`ReactiveMP.invoke_callback`](@ref) for more details.

See also: [`ReactiveMP.compute_product_of_messages`](@ref), [`ReactiveMP.compute_product_of_two_messages`]
"""
Base.@kwdef struct MessageProductContext{C, F, S, L, A}
    prod_constraint::C = BayesBase.GenericProd()
    form_constraint::F = UnspecifiedFormConstraint()
    form_constraint_check_strategy::S = FormConstraintCheckLast()
    fold_strategy::L = MessagesProductFromLeftToRight()
    callbacks::A = nothing
end

"""
    compute_product_of_two_messages(variable::AbstractVariable, context::MessageProductContext, left::Message, right::Message)

Computes the product of two messages `left` and `right` for a given `variable` using the provided `context`.
Returns a new message with the result of the multiplication (not necessarily normalized).
Applies `context.form_constraint` if `context.form_constraint_check_strategy` is set to [`ReactiveMP.FormConstraintCheckEach`](@ref).

The `variable` argument identifies which variable this product is being computed for, which is useful for callbacks (see [`ReactiveMP.BeforeProductOfTwoMessages`](@ref)).

## `is_clamped` and `is_initial`

The [`ReactiveMP.Message`](@ref) carries the `is_clamped` and `is_initial` flags.
The rules for the product are the following:
- If both messages are clamped, the result is clamped, OR
- If both messages are either clamped or initial, the result is initial, OR
- The result is neither clamped nor initial

See: [`ReactiveMP.MessageProductContext`](@ref), [`ReactiveMP.compute_product_of_messages`](@ref)
"""
function compute_product_of_two_messages(
    variable::AbstractVariable,
    context::MessageProductContext,
    left::Message,
    right::Message,
)
    invoke_callback(context.callbacks, BeforeProductOfTwoMessages(), variable, context, left, right)

    # We propagate clamped message, in case if both are clamped
    is_prod_clamped = is_clamped(left) && is_clamped(right)
    # We propagate initial message, in case if both are initial or left is initial and right is clameped or vice-versa
    is_prod_initial =
        !is_prod_clamped &&
        (is_clamped_or_initial(left)) &&
        (is_clamped_or_initial(right))

    # process distributions
    left_dist  = getdata(left)
    right_dist = getdata(right)
    new_dist   = prod(context.prod_constraint, left_dist, right_dist)

    if context.form_constraint_check_strategy === FormConstraintCheckEach()
        new_dist = constrain_form(context.form_constraint, new_dist)
    end

    # process addons
    left_addons  = getaddons(left)
    right_addons = getaddons(right)

    # process addons
    new_addons = multiply_addons(
        left_addons, right_addons, new_dist, left_dist, right_dist
    )
    result = Message(new_dist, is_prod_clamped, is_prod_initial, new_addons)

    invoke_callback(context.callbacks, AfterProductOfTwoMessages(), variable, context, left, right, result, new_addons)

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
    compute_product_of_messages(variable::AbstractVariable, context::MessageProductContext, messages)

Computes the product of a **collection** of messages for a given `variable` (as opposed to [`ReactiveMP.compute_product_of_two_messages`](@ref), which handles exactly **two** messages). Uses `context.fold_strategy` to determine the order in which [`ReactiveMP.compute_product_of_two_messages`](@ref) is called. By default this is [`ReactiveMP.MessagesProductFromLeftToRight`](@ref), but can be set to an arbitrary function that accepts `variable`, `context` and `messages` and which **must** call [`ReactiveMP.compute_product_of_two_messages`](@ref) under the hood.

See also: [`ReactiveMP.compute_product_of_two_messages`](@ref), [`ReactiveMP.MessagesProductFromLeftToRight`](@ref)
"""
function compute_product_of_messages(
    variable::AbstractVariable, context::MessageProductContext, messages
)
    result = as_message(
        compute_product_of_messages(
            context.fold_strategy, variable, context, messages
        ),
    )

    if context.form_constraint_check_strategy === FormConstraintCheckLast()
        result = Message(
            constrain_form(context.form_constraint, getdata(result)),
            is_clamped(result),
            is_initial(result),
            getaddons(result),
        )
    end

    return result
end

"""
    MessagesProductFromLeftToRight()

The default fold strategy for [`ReactiveMP.MessageProductContext`](@ref). Computes the product of messages from left to right using `foldl` within [`ReactiveMP.compute_product_of_messages`](@ref).
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

Alternative fold strategy for [`ReactiveMP.MessageProductContext`](@ref). Computes the product of messages from right to left using `foldr` within [`ReactiveMP.compute_product_of_messages`](@ref).
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

Custom fold strategy for [`ReactiveMP.compute_product_of_messages`](@ref). When `context.fold_strategy` is set to a `Function`,
it will be called with `variable`, `context` and `messages` as arguments. The function must call
[`ReactiveMP.compute_product_of_two_messages`](@ref) under the hood to compute the pairwise products.
"""
function compute_product_of_messages(
    f::Function,
    variable::AbstractVariable,
    context::MessageProductContext,
    messages,
)
    return f(variable, context, messages)
end

Distributions.pdf(message::Message, x)    = Distributions.pdf(getdata(message), x)
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
A special type of a message, for which the actual message is not computed immediately, but is computed later on demand (potentially never).
To compute and get the actual message, one needs to call the `as_message` method.
"""
mutable struct DeferredMessage{R, S, F} <: AbstractMessage
    const messages  :: R
    const marginals :: S
    const mappingFn :: F
    cache           :: Union{Nothing, Message}
end

DeferredMessage(messages::R, marginals::S, mappingFn::F) where {R, S, F} =
    DeferredMessage(messages, marginals, mappingFn, nothing)

function Base.show(io::IO, message::DeferredMessage)
    cache = getcache(message)
    if isnothing(cache)
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

dropproxytype(::Type{<:Message{T}}) where {T} = T

## Message observable 

struct MessageObservable{M <: AbstractMessage} <: Subscribable{M}
    subject :: Rocket.RecentSubjectInstance{M, Subject{M, AsapScheduler, AsapScheduler}}
    stream  :: LazyObservable{M}
end

MessageObservable(::Type{M} = AbstractMessage) where {M} =
    MessageObservable{M}(RecentSubject(M), lazy(M))

Rocket.getrecent(observable::MessageObservable) =
    Rocket.getrecent(observable.subject)

@inline Rocket.on_subscribe!(observable::MessageObservable, actor) =
    subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.Actor{<:AbstractMessage})           = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.NextActor{<:AbstractMessage})       = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.ErrorActor{<:AbstractMessage})      = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.CompletionActor{<:AbstractMessage}) = Rocket.on_subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.Subject{<:AbstractMessage})                 = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.BehaviorSubjectInstance{<:AbstractMessage}) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.PendingSubjectInstance{<:AbstractMessage})  = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.RecentSubjectInstance{<:AbstractMessage})   = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.ReplaySubjectInstance{<:AbstractMessage})   = Rocket.on_subscribe!(observable.stream, actor)

function connect!(message::MessageObservable, source)
    set!(message.stream, source |> multicast(message.subject) |> ref_count())
    return nothing
end

function setmessage!(message::MessageObservable, value)
    next!(message.subject, Message(value, false, true, nothing))
    return nothing
end

## Message Mapping structure
## https://github.com/JuliaLang/julia/issues/42559
## Explanation: Julia cannot fully infer type of the lambda callback function in activate! method in node.jl file
## We create a lambda-like callable structure to improve type inference and make it more stable
## However it is not fully inferrable due to dynamic tags and variable constraints, but still better than just a raw lambda callback
"""
    MessageMapping

A callable structure representing a deferred computation of a message in the
variational message passing framework. It stores all contextual information
necessary to compute a message later, such as variable tags, constraints,
addons, and the associated factor node.

`MessageMapping` replaces the original lambda-based implementation to improve
type stability and inference. When invoked as a function, it computes an
outgoing `Message` from given input messages and marginals using the appropriate
`@rule`.

See also: [`Message`](@ref), [`DeferredMessage`](@ref)
"""
struct MessageMapping{F, T, C, N, M, A, X, R, K, E}
    vtag            :: T
    vconstraint     :: C
    msgs_names      :: N
    marginals_names :: M
    meta            :: A
    addons          :: X
    factornode      :: R
    rulefallback    :: K
    callbacks       :: E
end

message_mapping_fform(::MessageMapping{F}) where {F} = F
message_mapping_fform(::MessageMapping{F}) where {F <: Function} = F.instance

# Some addons add post rule execution logic
function message_mapping_addons(
    mapping::MessageMapping, messages, marginals, result, addons
)
    return message_mapping_addons(
        mapping, mapping.addons, messages, marginals, result, addons
    )
end

# `enabled_addons` are always type-stable, whether `addons` are not, so we check based on the `enabled_addons` and ignore the `addons`
# As a consequence if any message update rule returns non-empty `addons`, but `enabled_addons` is empty, then the resulting value 
# of the `addons` will be simply ignored
message_mapping_addons(
    mapping::MessageMapping,
    enabled_addons::Nothing,
    messages,
    marginals,
    result,
    addons,
) = enabled_addons
message_mapping_addons(
    mapping::MessageMapping,
    enabled_addons::Tuple{},
    messages,
    marginals,
    result,
    addons,
) = enabled_addons

# The main logic here is that some addons may add extra computation AFTER the rule has been computed
# The benefit of that is that we have an access to the `MessageMapping` structure and is mostly useful for debug addons
function message_mapping_addons(
    mapping::MessageMapping,
    enabled_addons::Tuple,
    messages,
    marginals,
    result,
    addons,
)
    return map(addons) do addon
        return message_mapping_addon(
            addon, mapping, messages, marginals, result
        )
    end
end

# By default `message_mapping_addon` does nothing and simply returns the addon itself
# Other addons may override this behaviour (if necessary, see e.g. AddonMemory)
message_mapping_addon(addon, mapping, messages, marginals, result) = addon

function MessageMapping(
    ::Type{F},
    vtag::T,
    vconstraint::C,
    msgs_names::N,
    marginals_names::M,
    meta::A,
    addons::X,
    factornode::R,
    rulefallback::K,
    callbacks::E,
) where {F, T, C, N, M, A, X, R, K, E}
    return MessageMapping{F, T, C, N, M, A, X, R, K, E}(
        vtag,
        vconstraint,
        msgs_names,
        marginals_names,
        meta,
        addons,
        factornode,
        rulefallback,
        callbacks,
    )
end

function MessageMapping(
    ::F,
    vtag::T,
    vconstraint::C,
    msgs_names::N,
    marginals_names::M,
    meta::A,
    addons::X,
    factornode::R,
    rulefallback::K,
    callbacks::E,
) where {F <: Function, T, C, N, M, A, X, R, K, E}
    return MessageMapping{F, T, C, N, M, A, X, R, K, E}(
        vtag,
        vconstraint,
        msgs_names,
        marginals_names,
        meta,
        addons,
        factornode,
        rulefallback,
        callbacks,
    )
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

    invoke_callback(
        mapping.callbacks,
        BeforeMessageRuleCallback(),
        mapping,
        messages,
        marginals,
    )
    result, addons =
        if !isnothing(messages) &&
            any(ismissing, TupleTools.flatten(getdata.(messages)))
            missing, mapping.addons
        elseif !isnothing(marginals) &&
            any(ismissing, TupleTools.flatten(getdata.(marginals)))
            missing, mapping.addons
        else
            ruleargs = (
                message_mapping_fform(mapping),
                mapping.vtag,
                mapping.vconstraint,
                mapping.msgs_names,
                messages,
                mapping.marginals_names,
                marginals,
                mapping.meta,
                mapping.addons,
                mapping.factornode,
            )
            ruleoutput = rule(ruleargs...)
            # if `@rule` is not defined, the default behaviour is to return 
            # the `RuleMethodError` object
            if ruleoutput isa RuleMethodError
                if !isnothing(mapping.rulefallback)
                    mapping.rulefallback(ruleargs...)
                else
                    throw(ruleoutput)
                end
            else
                ruleoutput
            end
        end

    # Inject extra addons after the rule has been executed
    addons = message_mapping_addons(
        mapping, getdata(messages), getdata(marginals), result, addons
    )
    invoke_callback(
        mapping.callbacks,
        AfterMessageRuleCallback(),
        mapping,
        messages,
        marginals,
        result,
        addons,
    )

    return Message(result, is_message_clamped, is_message_initial, addons)
end
