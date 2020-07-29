export Message, Marginal
export getdata
export multiply_messages
export DefaultMessageGate, LoggerMessageGate, TransformMessageGate, MessageGatesComposition

import Base: *, +

using Distributions
using Rocket


struct Message{D}
    data :: D
end

struct Marginal{D}
    data :: D
end

getdata(message::Message)   = message.data
getdata(marginal::Marginal) = marginal.data

## Message

function multiply_messages end

Base.:*(m1::Message, m2::Message) = multiply_messages(m1, m2)

Distributions.mean(message::Message) = Distributions.mean(getdata(message))
Distributions.var(message::Message)  = Distributions.var(getdata(message))
Distributions.std(message::Message)  = Distributions.std(getdata(message))

Distributions.mean(message::Message{T}) where { T <: Real } = getdata(message)
Distributions.var(message::Message{T}) where { T <: Real }  = zero(T)
Distributions.std(message::Message{T}) where { T <: Real }  = zero(T)

as_message(data)                       = Message(data)
as_message(message::Message)   = message
as_message(marginal::Marginal) = Message(getdata(marginal))

## Marginal

Distributions.mean(marginal::Marginal) = Distributions.mean(getdata(marginal))
Distributions.var(marginal::Marginal)  = Distributions.var(getdata(marginal))
Distributions.std(marginal::Marginal)  = Distributions.std(getdata(marginal))

Distributions.mean(marginal::Marginal{T}) where { T <: Real } = getdata(marginal)
Distributions.var(marginal::Marginal{T}) where { T <: Real }  = zero(T)
Distributions.std(marginal::Marginal{T}) where { T <: Real }  = zero(T)

as_marginal(data)                       = Marginal(data)
as_marginal(marginal::Marginal) = marginal
as_marginal(message::Message)   = Marginal(getdata(message))

## Operators

reduce_messages(messages) = reduce(*, messages; init = Message(nothing))

const __as_message_operator  = Rocket.map(Message, as_message)
const __as_marginal_operator = Rocket.map(Marginal, as_marginal)

as_message()  = __as_message_operator
as_marginal() = __as_marginal_operator

const reduce_to_message  = Rocket.map(Message, (messages) -> reduce_messages(messages))
const reduce_to_marginal = Rocket.map(Marginal, (messages) -> as_marginal(reduce_messages(messages)))

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
