export AbstractMessage, Message, VariationalMessage
export getdata, is_clamped, is_initial, as_message
export multiply_messages

using Distributions
using Rocket

import Rocket: getrecent
import Base: ==, *, +, ndims, precision, length, size, show, nameof

"""
    AbstractMessage

An abstract supertype for all concrete message types.

See also: [`Message`](@ref)
"""
abstract type AbstractMessage end

"""
    materialize!(message::AbstractMessage)

Materializes an abstract message and converts it to be of type `Message`.

See also: [`Message`](@ref)
"""
function materialize! end

"""
    Message{D, A} <: AbstractMessage

`Message` structure encodes a **Belief Propagation** message, which holds some `data` that usually a probability distribution, but can also be an arbitrary object.
Message acts as a proxy structure to `data` object and proxies most of the statistical functions, e.g. `mean`, `mode`, `cov` etc.

# Arguments
- `data::D`: message always holds some data object associated with it
- `is_clamped::Bool`, specifies if this message is clamped
- `is_initial::Bool`, specifies if this message is initial
- `addons::A`, specifies the addons of the message

# Example 

```jldoctest
julia> distribution = Gamma(10.0, 2.0)
Gamma{Float64}(α=10.0, θ=2.0)

julia> message = Message(distribution, false, true, nothing)
Message(Gamma{Float64}(α=10.0, θ=2.0))

julia> mean(message) 
20.0

julia> getdata(message)
Gamma{Float64}(α=10.0, θ=2.0)

julia> is_clamped(message)
false

julia> is_initial(message)
true

```

See also: [`AbstractMessage`](@ref), [`ReactiveMP.materialize!`](@ref)
"""
struct Message{D, A} <: AbstractMessage
    data       :: D
    is_clamped :: Bool
    is_initial :: Bool
    addons     :: A
end

"""
    getdata(message::Message)    

Returns `data` associated with the `message`.
"""
getdata(message::Message) = message.data

"""
    is_clamped(message::Message)

Checks if `message` is clamped or not.

See also: [`is_initial`](@ref)
"""
is_clamped(message::Message) = message.is_clamped

"""
    is_initial(message::Message)

Checks if `message` is initial or not.

See also: [`is_clamped`](@ref)
"""
is_initial(message::Message) = message.is_initial
getaddons(message::Message) = message.addons

typeofdata(message::Message) = typeof(getdata(message))

getdata(messages::NTuple{N, <:Message}) where {N} = map(getdata, messages)
getdata(messages::AbstractArray{<:Message})       = map(getdata, messages)

materialize!(message::Message) = message

# Base.show(io::IO, message::Message) = print(io, string("Message(", getdata(message), ") with ", string(getaddons(message))))
function show(io::IO, message::Message)
    print(io, string("Message(", getdata(message), ")"))
    if !isnothing(getaddons(message))
        print(io, ") with ", string(getaddons(message)))
    end
end

Base.:*(left::Message, right::Message) = multiply_messages(ProdAnalytical(), left, right)

# We need this dummy method as Julia is not smart enough to 
# do that automatically if `data` is mutable
function Base.:(==)(left::Message, right::Message)
    return left.is_clamped == right.is_clamped && left.is_initial == right.is_initial && left.data == right.data && left.addons == right.addons
end

function multiply_messages(prod_constraint, left::Message, right::Message)
    # We propagate clamped message, in case if both are clamped
    is_prod_clamped = is_clamped(left) && is_clamped(right)
    # We propagate initial message, in case if both are initial or left is initial and right is clameped or vice-versa
    is_prod_initial = !is_prod_clamped && (is_clamped_or_initial(left)) && (is_clamped_or_initial(right))

    # process distributions
    left_dist  = getdata(left)
    right_dist = getdata(right)
    new_dist   = prod(prod_constraint, left_dist, right_dist)

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
# We want to cast it explicitly to a Message structure (which as_message does in case of VariationalMessage)
# We use with Base.Generator to reduce an amount of memory used by this procedure since Generator generates items lazily
prod_foldl_reduce(prod_constraint, form_constraint, ::FormConstraintCheckEach) =
    (messages) -> foldl((left, right) -> constrain_form_as_message(multiply_messages(prod_constraint, left, right), form_constraint), Base.Generator(as_message, messages))

prod_foldl_reduce(prod_constraint, form_constraint, ::FormConstraintCheckLast) =
    (messages) -> constrain_form_as_message(foldl((left, right) -> multiply_messages(prod_constraint, left, right), Base.Generator(as_message, messages)), form_constraint)

prod_foldr_reduce(prod_constraint, form_constraint, ::FormConstraintCheckEach) =
    (messages) -> foldr((left, right) -> constrain_form_as_message(multiply_messages(prod_constraint, left, right), form_constraint), Base.Generator(as_message, messages))

prod_foldr_reduce(prod_constraint, form_constraint, ::FormConstraintCheckLast) =
    (messages) -> constrain_form_as_message(foldr((left, right) -> multiply_messages(prod_constraint, left, right), Base.Generator(as_message, messages)), form_constraint)

Distributions.pdf(message::Message, x)    = Distributions.pdf(getdata(message), x)
Distributions.logpdf(message::Message, x) = Distributions.logpdf(getdata(message), x)

MacroHelpers.@proxy_methods Message getdata [
    Distributions.mean,
    Distributions.median,
    Distributions.mode,
    Distributions.shape,
    Distributions.scale,
    Distributions.rate,
    Distributions.var,
    Distributions.std,
    Distributions.cov,
    Distributions.invcov,
    Distributions.logdetcov,
    Distributions.entropy,
    Distributions.params,
    Base.precision,
    Base.length,
    Base.ndims,
    Base.size,
    Base.eltype,
    mean_cov,
    mean_var,
    mean_invcov,
    mean_precision,
    weightedmean_cov,
    weightedmean_var,
    weightedmean_invcov,
    weightedmean_precision,
    probvec,
    weightedmean
]

Distributions.mean(fn::Function, message::Message) = mean(fn, getdata(message))

## Variational Message

mutable struct VariationalMessage{R, S, F} <: AbstractMessage
    messages  :: R
    marginals :: S
    mappingFn :: F
    cache     :: Union{Nothing, Message}
end

VariationalMessage(messages::R, marginals::S, mappingFn::F) where {R, S, F} = VariationalMessage(messages, marginals, mappingFn, nothing)

Base.show(io::IO, ::VariationalMessage) = print(io, "VariationalMessage()")

getcache(vmessage::VariationalMessage)                    = vmessage.cache
setcache!(vmessage::VariationalMessage, message::Message) = vmessage.cache = message

function materialize!(vmessage::VariationalMessage)
    cache = getcache(vmessage)
    if cache !== nothing
        return cache
    end
    message = materialize!(vmessage.mappingFn, (getrecent(vmessage.messages), getrecent(vmessage.marginals)))
    setcache!(vmessage, message)
    return message
end

## Utility functions

as_message(message::Message)             = message
as_message(vmessage::VariationalMessage) = materialize!(vmessage)

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

struct MessageMapping{F, T, C, N, M, A, X, R}
    vtag            :: T
    vconstraint     :: C
    msgs_names      :: N
    marginals_names :: M
    meta            :: A
    addons          :: X
    factornode      :: R
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

function MessageMapping(::Type{F}, vtag::T, vconstraint::C, msgs_names::N, marginals_names::M, meta::A, addons::X, factornode::R) where {F, T, C, N, M, A, X, R}
    return MessageMapping{F, T, C, N, M, A, X, R}(vtag, vconstraint, msgs_names, marginals_names, meta, addons, factornode)
end

function MessageMapping(::F, vtag::T, vconstraint::C, msgs_names::N, marginals_names::M, meta::A, addons::X, factornode::R) where {F <: Function, T, C, N, M, A, X, R}
    return MessageMapping{F, T, C, N, M, A, X, R}(vtag, vconstraint, msgs_names, marginals_names, meta, addons, factornode)
end

function materialize!(mapping::MessageMapping, dependencies)
    return materialize!(mapping, dependencies[1], dependencies[2])
end

function materialize!(mapping::MessageMapping, messages, marginals)
    # Message is clamped if all of the inputs are clamped
    is_message_clamped = __check_all(is_clamped, messages) && __check_all(is_clamped, marginals)

    # Message is initial if it is not clamped and all of the inputs are either clamped or initial
    is_message_initial = !is_message_clamped && (__check_all(is_clamped_or_initial, messages) && __check_all(is_clamped_or_initial, marginals))

    result, addons = rule(
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

    # Inject extra addons after the rule has been executed
    addons = message_mapping_addons(mapping, getdata(messages), getdata(marginals), result, addons)

    return Message(result, is_message_clamped, is_message_initial, addons)
end
