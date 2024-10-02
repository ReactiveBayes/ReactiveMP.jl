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
    return left.is_clamped == right.is_clamped && left.is_initial == right.is_initial && left.data == right.data && left.addons == right.addons
end

"""
    multiply_messages(prod_strategy, left::Message, right::Message)

Multiplies two messages `left` and `right` using a given product strategy `prod_strategy`.
Returns a new message with the result of the multiplication. Note that the resulting message is not necessarily normalized.
"""
function multiply_messages(prod_strategy, left::Message, right::Message)
    # We propagate clamped message, in case if both are clamped
    is_prod_clamped = is_clamped(left) && is_clamped(right)
    # We propagate initial message, in case if both are initial or left is initial and right is clameped or vice-versa
    is_prod_initial = !is_prod_clamped && (is_clamped_or_initial(left)) && (is_clamped_or_initial(right))

    # process distributions
    left_dist  = getdata(left)
    right_dist = getdata(right)
    new_dist   = prod(prod_strategy, left_dist, right_dist)

    # process addons
    left_addons  = getaddons(left)
    right_addons = getaddons(right)

    # process addons
    new_addons = multiply_addons(left_addons, right_addons, new_dist, left_dist, right_dist)

    return Message(new_dist, is_prod_clamped, is_prod_initial, new_addons)
end

constrain_form_as_message(message::Message, form_constraint) =
    Message(constrain_form(form_constraint, getdata(message)), is_clamped(message), is_initial(message), getaddons(message))

# Note: we need extra Base.Generator(as_message, messages) step here, because some of the messages might be VMP messages
# We want to cast it explicitly to a Message structure (which as_message does in case of DefferedMessage)
# We use with Base.Generator to reduce an amount of memory used by this procedure since Generator generates items lazily
prod_foldl_reduce(prod_constraint, form_constraint, ::FormConstraintCheckEach) =
    (messages) -> foldl((left, right) -> constrain_form_as_message(multiply_messages(prod_constraint, left, right), form_constraint), Base.Generator(as_message, messages))

prod_foldl_reduce(prod_constraint, form_constraint, ::FormConstraintCheckLast) =
    (messages) -> constrain_form_as_message(foldl((left, right) -> multiply_messages(prod_constraint, left, right), Base.Generator(as_message, messages)), form_constraint)

prod_foldr_reduce(prod_constraint, form_constraint, ::FormConstraintCheckEach) =
    (messages) -> foldr((left, right) -> constrain_form_as_message(multiply_messages(prod_constraint, left, right), form_constraint), Base.Generator(as_message, messages))

prod_foldr_reduce(prod_constraint, form_constraint, ::FormConstraintCheckLast) =
    (messages) -> constrain_form_as_message(foldr((left, right) -> multiply_messages(prod_constraint, left, right), Base.Generator(as_message, messages)), form_constraint)

# Base.:*(m1::Message, m2::Message) = multiply_messages(m1, m2)

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
    Base.eltype
]

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

DeferredMessage(messages::R, marginals::S, mappingFn::F) where {R, S, F} = DeferredMessage(messages, marginals, mappingFn, nothing)

function Base.show(io::IO, message::DeferredMessage)
    cache = getcache(message)
    if isnothing(cache)
        print(io, "DeferredMessage([ use `as_message` to compute the message ])")
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
    return as_message(message, cache, getrecent(message.messages), getrecent(message.marginals))
end

function as_message(message::DeferredMessage, cache::Nothing, messages, marginals)::Message
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

MessageObservable(::Type{M} = AbstractMessage) where {M} = MessageObservable{M}(RecentSubject(M), lazy(M))

Rocket.getrecent(observable::MessageObservable) = Rocket.getrecent(observable.subject)

@inline Rocket.on_subscribe!(observable::MessageObservable, actor) = subscribe!(observable.stream, actor)

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

struct MessageMapping{F, T, C, N, M, A, X, R, K}
    vtag            :: T
    vconstraint     :: C
    msgs_names      :: N
    marginals_names :: M
    meta            :: A
    addons          :: X
    factornode      :: R
    rulefallback    :: K
end

message_mapping_fform(::MessageMapping{F}) where {F} = F
message_mapping_fform(::MessageMapping{F}) where {F <: Function} = F.instance

# Some addons add post rule execution logic
function message_mapping_addons(mapping::MessageMapping, messages, marginals, result, addons)
    return message_mapping_addons(mapping, mapping.addons, messages, marginals, result, addons)
end

# `enabled_addons` are always type-stable, whether `addons` are not, so we check based on the `enabled_addons` and ignore the `addons`
# As a consequence if any message update rule returns non-empty `addons`, but `enabled_addons` is empty, then the resulting value 
# of the `addons` will be simply ignored
message_mapping_addons(mapping::MessageMapping, enabled_addons::Nothing, messages, marginals, result, addons) = enabled_addons
message_mapping_addons(mapping::MessageMapping, enabled_addons::Tuple{}, messages, marginals, result, addons) = enabled_addons

# The main logic here is that some addons may add extra computation AFTER the rule has been computed
# The benefit of that is that we have an access to the `MessageMapping` structure and is mostly useful for debug addons
function message_mapping_addons(mapping::MessageMapping, enabled_addons::Tuple, messages, marginals, result, addons)
    return map(addons) do addon
        return message_mapping_addon(addon, mapping, messages, marginals, result)
    end
end

# By default `message_mapping_addon` does nothing and simply returns the addon itself
# Other addons may override this behaviour (if necessary, see e.g. AddonMemory)
message_mapping_addon(addon, mapping, messages, marginals, result) = addon

function MessageMapping(::Type{F}, vtag::T, vconstraint::C, msgs_names::N, marginals_names::M, meta::A, addons::X, factornode::R, rulefallback::K) where {F, T, C, N, M, A, X, R, K}
    return MessageMapping{F, T, C, N, M, A, X, R, K}(vtag, vconstraint, msgs_names, marginals_names, meta, addons, factornode, rulefallback)
end

function MessageMapping(
    ::F, vtag::T, vconstraint::C, msgs_names::N, marginals_names::M, meta::A, addons::X, factornode::R, rulefallback::K
) where {F <: Function, T, C, N, M, A, X, R, K}
    return MessageMapping{F, T, C, N, M, A, X, R, K}(vtag, vconstraint, msgs_names, marginals_names, meta, addons, factornode, rulefallback)
end

function (mapping::MessageMapping)(messages, marginals)
    # Message is clamped if all of the inputs are clamped
    is_message_clamped = __check_all(is_clamped, messages) && __check_all(is_clamped, marginals)

    # Message is initial if it is not clamped and all of the inputs are either clamped or initial
    is_message_initial = !is_message_clamped && (__check_all(is_clamped_or_initial, messages) && __check_all(is_clamped_or_initial, marginals))

    result, addons = if !isnothing(messages) && any(ismissing, TupleTools.flatten(getdata.(messages)))
        missing, mapping.addons
    elseif !isnothing(marginals) && any(ismissing, TupleTools.flatten(getdata.(marginals)))
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
            mapping.factornode
        )
        ruleoutput = rule(ruleargs...)
        # if `@rule` is not defined, the default behaviour is to return 
        # the `RuleMethodError` object
        if ruleoutput isa RuleMethodError
            !isnothing(mapping.rulefallback) ? mapping.rulefallback(ruleargs...) : throw(ruleoutput)
        else
            ruleoutput
        end
    end

    # Inject extra addons after the rule has been executed
    addons = message_mapping_addons(mapping, getdata(messages), getdata(marginals), result, addons)

    return Message(result, is_message_clamped, is_message_initial, addons)
end
