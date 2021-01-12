export Message, getdata, as_message
export multiply_messages

using Distributions
using Rocket

import Base: *, +, ndims, precision, length, size, show

struct Message{D}
    data :: D
end

getdata(message::Message)                          = message.data
getdata(messages::NTuple{ N, <: Message }) where N = map(getdata, messages)

Base.show(io::IO, message::Message) = print(io, string("Message(", getdata(message), ")"))

mutable struct DefferedMessageProps
    cache :: Any
end

struct DefferedMessage{R, F}
    dependencies :: R
    mappingFn    :: F
    props        :: DefferedMessageProps
end

DefferedMessage(deps::R, mappingFn::F) where { R, F } = DefferedMessage(deps, mappingFn, DefferedMessageProps(nothing))

function getdata(message::DefferedMessage)
    if message.props.cache !== nothing
        return message.props.cache
    end
    message.props.cache = message.mappingFn(message.dependencies)
    return message.props.cache
end

## Message

multiply_messages(left::Message, right::Message) = as_message(prod(ProdPreserveParametrisation(), getdata(left), getdata(right)))

Base.:*(m1::Message, m2::Message)         = multiply_messages(m1, m2)
Base.:*(m1::Message, m2::DefferedMessage) = multiply_messages(m1, as_message(getdata(m2)))

Distributions.mean(message::Message)      = Distributions.mean(getdata(message))
Distributions.median(message::Message)    = Distributions.median(getdata(message))
Distributions.mode(message::Message)      = Distributions.mode(getdata(message))
Distributions.var(message::Message)       = Distributions.var(getdata(message))
Distributions.std(message::Message)       = Distributions.std(getdata(message))
Distributions.cov(message::Message)       = Distributions.cov(getdata(message))
Distributions.invcov(message::Message)    = Distributions.invcov(getdata(message))
Distributions.logdetcov(message::Message) = Distributions.logdetcov(getdata(message))
Distributions.entropy(message::Message)   = Distributions.entropy(getdata(message))

Distributions.pdf(message::Message, x)    = Distributions.pdf(getdata(message), x)
Distributions.logpdf(message::Message, x) = Distributions.logpdf(getdata(message), x)

Base.precision(message::Message) = precision(getdata(message))
Base.length(message::Message)    = length(getdata(message))
Base.ndims(message::Message)     = ndims(getdata(message))
Base.size(message::Message)      = size(getdata(message))

probvec(message::Message)         = probvec(getdata(message))
logmean(message::Message)         = logmean(getdata(message))
inversemean(message::Message)     = inversemean(getdata(message))
mirroredlogmean(message::Message) = mirroredlogmean(getdata(message))

## Utility functions

as_message(data)               = Message(data)
as_message(message::Message)   = message

## Operators

reduce_messages(messages) = reduce(*, messages; init = Message(missing))

const __as_message_operator  = Rocket.map(Message, as_message)

as_message()  = __as_message_operator

function __reduce_to_message(messages)
    return as_message(reduce_messages(messages))
end

const reduce_to_message  = Rocket.map(Message, __reduce_to_message)
