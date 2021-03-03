export Message, getdata, as_message
export multiply_messages

using Distributions
using Rocket

import Base: *, +, ndims, precision, length, size, show

abstract type AbstractMessage end

struct Message{D} <: AbstractMessage
    data :: D
end

getdata(message::Message)                          = message.data
getdata(messages::NTuple{ N, <: Message }) where N = map(getdata, messages)

materialize!(message::Message) = message

Base.show(io::IO, message::Message) = print(io, string("Message(", getdata(message), ")"))

## Message

multiply_messages(left::Message, right::Message) = as_message(prod(ProdPreserveParametrisation(), getdata(left), getdata(right)))

Base.:*(m1::Message, m2::Message) = multiply_messages(m1, m2)

Distributions.mean(message::Message)      = Distributions.mean(getdata(message))
Distributions.median(message::Message)    = Distributions.median(getdata(message))
Distributions.mode(message::Message)      = Distributions.mode(getdata(message))
Distributions.shape(message::Message)     = Distributions.shape(getdata(message))
Distributions.scale(message::Message)     = Distributions.scale(getdata(message))
Distributions.rate(message::Message)     = Distributions.rate(getdata(message))
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
logmean(message::Message)         = logmean(getdata(message))
inversemean(message::Message)     = inversemean(getdata(message))
mirroredlogmean(message::Message) = mirroredlogmean(getdata(message))

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

Base.show(io::IO, message::VariationalMessage) = print(io, string("VariationalMessage(:postponed)"))

getcache(vmessage::VariationalMessage)                    = vmessage.props.cache
setcache!(vmessage::VariationalMessage, message::Message) = vmessage.props.cache = message

compute_message(vmessage::VariationalMessage) = as_message(vmessage.mappingFn((vmessage.messages, getrecent(vmessage.marginals))))

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

as_message(data)                         = Message(data)
as_message(message::Message)             = message
as_message(vmessage::VariationalMessage) = materialize!(vmessage)

## Operators

reduce_messages(messages) = mapreduce(as_message, *, messages; init = Message(missing))

const __as_message_operator  = Rocket.map(Message, as_message)

as_message()  = __as_message_operator

function __reduce_to_message(messages)
    return as_message(reduce_messages(messages))
end

const reduce_to_message  = Rocket.map(Message, __reduce_to_message)
