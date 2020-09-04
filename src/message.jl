export Message, getdata, as_message
export multiply_messages
export DefaultMessageGate, LoggerMessageGate, TransformMessageGate, MessageGatesComposition

import Base: *, +, ndims, precision

using Distributions
using Rocket

struct Message{D}
    data :: D
end

getdata(message::Message)   = message.data

## Message

function multiply_messages end

Base.:*(m1::Message, m2::Message) = multiply_messages(m1, m2)

Distributions.mean(message::Message) = Distributions.mean(getdata(message))
Distributions.var(message::Message)  = Distributions.var(getdata(message))
Distributions.std(message::Message)  = Distributions.std(getdata(message))
Distributions.cov(message::Message)  = Distributions.cov(getdata(message))

Base.precision(message::Message) = precision(getdata(message))
Base.ndims(message::Message)     = ndims(getdata(message))

## Delta function message

Distributions.mean(message::Message{T}) where { T <: Real } = getdata(message)
Distributions.var(message::Message{T}) where { T <: Real }  = zero(T)
Distributions.std(message::Message{T}) where { T <: Real }  = zero(T)
Distributions.cov(message::Message{T}) where { T <: Real }  = zero(T)

Base.precision(message::Message{T}) where { T <: Real } = Inf
Base.ndims(message::Message{T})     where { T <: Real } = 1

logmean(message::Message{T}) where { T <: Real }     = log(getdata(message))
inversemean(message::Message{T}) where { T <: Real } = 1.0 / getdata(message)

## Vector-based delta function message

Distributions.mean(message::Message{T}) where { T <: Vector } = getdata(message)
Distributions.var(message::Message{T}) where { T <: Vector }  = zero(T)
Distributions.std(message::Message{T}) where { T <: Vector }  = zero(T)
Distributions.cov(message::Message{T}) where { T <: Vector }  = zero(T)

Base.precision(message::Message{T}) where { T <: Vector } = Inf
Base.ndims(message::Message{T})     where { T <: Vector } = length(getdata(message))

## Utility functions

as_message(data)               = Message(data)
as_message(message::Message)   = message

## Operators

reduce_messages(messages) = reduce(*, messages; init = Message(nothing))

const __as_message_operator  = Rocket.map(Message, as_message)

as_message()  = __as_message_operator

const reduce_to_message  = Rocket.map(Message, (messages) -> reduce_messages(messages))

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
