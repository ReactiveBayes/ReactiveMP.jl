export AbstractMessage, Message
export AbstractMarginal, Marginal
export getdata
export multiply_messages
export DefaultMessageGate, LoggerMessageGate, TransformMessageGate, MessageGatesComposition

import Base: *, +

using Distributions
using Rocket

## AbstractMessage

abstract type AbstractMessage end

## AbstractMarginal

abstract type AbstractMarginal end

## Message

struct Message{D} <: AbstractMessage
    data :: D
end

getdata(message::Message) = message.data

function multiply_messages end

Base.:*(m1::AbstractMessage, m2::AbstractMessage) = multiply_messages(m1, m2)

Distributions.mean(message::Message) = Distributions.mean(getdata(message))
Distributions.var(message::Message)  = Distributions.var(getdata(message))
Distributions.std(message::Message)  = Distributions.std(getdata(message))

Distributions.mean(message::Message{T}) where { T <: Real } = getdata(message)
Distributions.var(message::Message{T}) where { T <: Real }  = zero(T)
Distributions.std(message::Message{T}) where { T <: Real }  = zero(T)

as_message(data)                       = Message(data)
as_message(message::AbstractMessage)   = message
as_message(marginal::AbstractMarginal) = Message(getdata(marginal))

## Marginal

struct Marginal{D} <: AbstractMarginal
    data :: D
end

getdata(marginal::Marginal) = marginal.data

Distributions.mean(marginal::Marginal) = Distributions.mean(getdata(marginal))
Distributions.var(marginal::Marginal)  = Distributions.var(getdata(marginal))
Distributions.std(marginal::Marginal)  = Distributions.std(getdata(marginal))

Distributions.mean(marginal::Marginal{T}) where { T <: Real } = getdata(marginal)
Distributions.var(marginal::Marginal{T}) where { T <: Real }  = zero(T)
Distributions.std(marginal::Marginal{T}) where { T <: Real }  = zero(T)

as_marginal(data)                       = Marginal(data)
as_marginal(marginal::AbstractMarginal) = marginal
as_marginal(message::AbstractMessage)   = Marginal(getdata(message))

## Operators

reduce_messages(messages) = reduce(*, messages; init = Message(nothing))

const __as_message_operator  = Rocket.map(AbstractMessage, as_message)
const __as_marginal_operator = Rocket.map(AbstractMarginal, as_marginal)

as_message()  = __as_message_operator
as_marginal() = __as_marginal_operator

const reduce_to_message  = Rocket.map(AbstractMessage, (messages) -> reduce_messages(messages))
const reduce_to_marginal = Rocket.map(AbstractMarginal, (messages) -> as_marginal(reduce_messages(messages)))

## Gates

abstract type AbstractMessageGate end

struct DefaultMessageGate <: AbstractMessageGate end

gate!(::DefaultMessageGate, node, variable, message)  = message

struct LoggerMessageGate <: AbstractMessageGate end

function gate!(::LoggerMessageGate, node, variable, message)
    println(string("From variable ", variable, " of node ", functionalform(node), " => ", message));
    return message
end

struct TransformMessageGate{F} <: AbstractMessageGate
    transformFn::F
end

gate!(tg::TransformMessageGate, node, variable, message) = tg.transformFn(node, variable, message)

struct MessageGatesComposition{C} <: AbstractMessageGate
    composition :: C
end

gate!(gc::MessageGatesComposition, node, variable, message) = foldl((m, g) -> gate!(g, node, variable, m), gc.composition, init = message)

Base.:+(gate1::AbstractMessageGate,     gate2::AbstractMessageGate)     = MessageGatesComposition((gate1, gate2))
Base.:+(gate1::MessageGatesComposition, gate2::AbstractMessageGate)     = MessageGatesComposition((gate1.composition..., gate2))
Base.:+(gate1::AbstractMessageGate,     gate2::MessageGatesComposition) = MessageGatesComposition((gate1, gate2.composition...))
Base.:+(gate1::MessageGatesComposition, gate2::MessageGatesComposition) = MessageGatesComposition((gate1.composition..., gate2.composition...))

gate!(::AbstractMessageGate, node, variable, message) = gate!(DefaultMessageGate(), node, variable, message)
