export Message, getdata, as_message
export multiply_messages
export DefaultMessageGate, LoggerMessageGate, TransformMessageGate, MessageGatesComposition

using Distributions
using Rocket

import Base: *, +, ndims, precision, length, size

struct Message{D}
    data :: D
end

getdata(message::Message)   = message.data

## Message

function multiply_messages end

# TODO
multiply_messages(left::Message, right::Message) = as_message(prod(ProdPreserveParametrisation(), getdata(left), getdata(right)))

Base.:*(m1::Message, m2::Message) = multiply_messages(m1, m2)

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

logmean(message::Message)         = logmean(getdata(message))
inversemean(message::Message)     = inversemean(getdata(message))
mirroredlogmean(message::Message) = mirroredlogmean(getdata(message))

## Utiliy nothing message

function multiply_messages(m1::Message{Nothing}, m2::Message{Nothing})
    return Message(nothing)
end

@symmetrical function multiply_messages(::Message{Nothing}, message::Message)
    return message
end

## Utility functions

as_message(data)               = Message(data)
as_message(message::Message)   = message

## Operators

reduce_messages(messages) = reduce(*, messages; init = Message(nothing))

const __as_message_operator  = Rocket.map(Message, as_message)

as_message()  = __as_message_operator

function __reduce_to_message(messages)
    return as_message(reduce_messages(messages))
end

const reduce_to_message  = Rocket.map(Message, __reduce_to_message)

## Gates

abstract type MessageGate end

struct DefaultMessageGate <: MessageGate end

gate!(::DefaultMessageGate, node, variable, message)  = message

struct LoggerMessageGate <: MessageGate end

function gate!(::LoggerMessageGate, node, variable, message)
    println(string("From variable ", variable, " of node ", functionalform(node), " => ", message));
    return message
end

struct TransformMessageGate{F} <: MessageGate
    transformFn::F
end

gate!(tg::TransformMessageGate, node, variable, message) = tg.transformFn(node, variable, message)

struct MessageGatesComposition{C} <: MessageGate
    composition :: C
end

gate!(gc::MessageGatesComposition, node, variable, message) = foldl((m, g) -> gate!(g, node, variable, m), gc.composition, init = message)

Base.:+(gate1::MessageGate,     gate2::MessageGate)                     = MessageGatesComposition((gate1, gate2))
Base.:+(gate1::MessageGatesComposition, gate2::MessageGate)             = MessageGatesComposition((gate1.composition..., gate2))
Base.:+(gate1::MessageGate,     gate2::MessageGatesComposition)         = MessageGatesComposition((gate1, gate2.composition...))
Base.:+(gate1::MessageGatesComposition, gate2::MessageGatesComposition) = MessageGatesComposition((gate1.composition..., gate2.composition...))

gate!(::MessageGate, node, variable, message) = gate!(DefaultMessageGate(), node, variable, message)
