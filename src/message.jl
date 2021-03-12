export Message, getdata, is_clamped, is_initial, as_message
export multiply_messages

using Distributions
using Rocket

import Rocket: getrecent
import Base: *, +, ndims, precision, length, size, show

abstract type AbstractMessage end

struct Message{D} <: AbstractMessage
    data       :: D
    is_clamped :: Bool
    is_initial :: Bool
end

getdata(message::Message)    = message.data
is_clamped(message::Message) = message.is_clamped
is_initial(message::Message) = message.is_initial

getdata(messages::NTuple{ N, <: Message })    where N = map(getdata, messages)

materialize!(message::Message) = message

Base.show(io::IO, message::Message) = print(io, string("Message(", getdata(message), ")"))

## Message

function multiply_messages(left::Message, right::Message) 
    # We propagate clamped message, in case if both are clamped
    is_prod_clamped = is_clamped(left) && is_clamped(right)
    # We propagate initial message, in case if both are initial or left is initial and right is clameped or vice-versa
    is_prod_initial = !is_prod_clamped && (is_initial(left) || is_clamped(left)) && (is_initial(right) || is_clamped(right))

    return Message(prod(ProdPreserveParametrisation(), getdata(left), getdata(right)), is_prod_clamped, is_prod_initial)
end

Base.:*(m1::Message, m2::Message) = multiply_messages(m1, m2)

Distributions.mean(message::Message)      = Distributions.mean(getdata(message))
Distributions.median(message::Message)    = Distributions.median(getdata(message))
Distributions.mode(message::Message)      = Distributions.mode(getdata(message))
Distributions.shape(message::Message)     = Distributions.shape(getdata(message))
Distributions.scale(message::Message)     = Distributions.scale(getdata(message))
Distributions.rate(message::Message)      = Distributions.rate(getdata(message))
Distributions.var(message::Message)       = Distributions.var(getdata(message))
Distributions.std(message::Message)       = Distributions.std(getdata(message))
Distributions.cov(message::Message)       = Distributions.cov(getdata(message))
Distributions.invcov(message::Message)    = Distributions.invcov(getdata(message))
Distributions.logdetcov(message::Message) = Distributions.logdetcov(getdata(message))
Distributions.entropy(message::Message)   = Distributions.entropy(getdata(message))
Distributions.params(message::Message)    = Distributions.params(getdata(message))

Distributions.pdf(message::Message, x)    = Distributions.pdf(getdata(message), x)
Distributions.logpdf(message::Message, x) = Distributions.logpdf(getdata(message), x)

Base.precision(message::Message) = precision(getdata(message))
Base.length(message::Message)    = length(getdata(message))
Base.ndims(message::Message)     = ndims(getdata(message))
Base.size(message::Message)      = size(getdata(message))

probvec(message::Message)         = probvec(getdata(message))
weightedmean(message::Message)    = weightedmean(getdata(message))
inversemean(message::Message)     = inversemean(getdata(message))
logmean(message::Message)         = logmean(getdata(message))
meanlogmean(message::Message)     = meanlogmean(getdata(message))
mirroredlogmean(message::Message) = mirroredlogmean(getdata(message))
loggammamean(message::Message)    = loggammamean(getdata(message))

## Variational Message

mutable struct VariationalMessageProps
    cache :: Union{Nothing, Message}
end

struct VariationalMessage{R, S, F} <: AbstractMessage
    messages   :: R
    marginals  :: S
    mappingFn  :: F
    props      :: VariationalMessageProps
end

VariationalMessage(messages::R, marginals::S, mappingFn::F) where { R, S, F } = VariationalMessage(messages, marginals, mappingFn, VariationalMessageProps(nothing))

Base.show(io::IO, ::VariationalMessage) = print(io, string("VariationalMessage(:postponed)"))

getcache(vmessage::VariationalMessage)                    = vmessage.props.cache
setcache!(vmessage::VariationalMessage, message::Message) = vmessage.props.cache = message

__check_all(fn::Function, iterator)  = all(fn, iterator)
__check_all(fn::Function, ::Nothing) = true

compute_message(vmessage::VariationalMessage) = vmessage.mappingFn((vmessage.messages, getrecent(vmessage.marginals)))

function materialize!(vmessage::VariationalMessage)
    cache = getcache(vmessage)
    if cache !== nothing
        return cache
    end
    message = compute_message(vmessage)
    setcache!(vmessage, message)
    return message
end

## Utility functions

as_message(message::Message)             = message
as_message(vmessage::VariationalMessage) = materialize!(vmessage)

## Operators

reduce_messages(messages) = mapreduce(as_message, *, messages)
