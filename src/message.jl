export AbstractMessage, Message
export AbstractBelief, Belief
export getdata
export multiply_messages

import Base: *

using Distributions
using Rocket

## AbstractMessage

abstract type AbstractMessage end

## AbstractBelief

abstract type AbstractBelief end

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

as_message(data)                     = Message(data)
as_message(message::AbstractMessage) = message
as_message(belief::AbstractBelief)   = Message(getdata(belief))

## Belief

struct Belief{D} <: AbstractBelief
    data :: D
end

getdata(belief::Belief) = belief.data

Distributions.mean(belief::Belief) = Distributions.mean(getdata(belief))
Distributions.var(belief::Belief)  = Distributions.var(getdata(belief))
Distributions.std(belief::Belief)  = Distributions.std(getdata(belief))

Distributions.mean(belief::Belief{T}) where { T <: Real } = getdata(belief)
Distributions.var(belief::Belief{T}) where { T <: Real }  = zero(T)
Distributions.std(belief::Belief{T}) where { T <: Real }  = zero(T)

as_belief(data)                     = Belief(data)
as_belief(belief::AbstractBelief)   = belief
as_belief(message::AbstractMessage) = Belief(getdata(message))

## Operators

reduce_messages(messages) = reduce(*, messages; init = Message(nothing))

const __as_message_operator = Rocket.map(AbstractMessage, as_message)
const __as_belief_operator  = Rocket.map(AbstractBelief, as_belief)

as_message() = __as_message_operator
as_belief()  = __as_belief_operator

const reduce_to_message = Rocket.map(AbstractMessage, (messages) -> reduce_messages(messages))
const reduce_to_belief  = Rocket.map(AbstractBelief, (messages) -> as_belief(reduce_messages(messages)))
