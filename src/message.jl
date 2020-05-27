export AbstractMessage, Message, multiply_messages, reduce_messages
export AbstractBelief, Belief
export getdata

import Base: *
using Distributions

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

function reduce_messages(messages)
    return reduce(*, messages; init = Message(nothing))
end

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

function reduce_message_to_belief(messages)
    return as_belief(reduce_messages(messages))
end

Distributions.mean(belief::Belief) = Distributions.mean(getdata(belief))
Distributions.var(belief::Belief)  = Distributions.var(getdata(belief))
Distributions.std(belief::Belief)  = Distributions.std(getdata(belief))

Distributions.mean(belief::Belief{T}) where { T <: Real } = getdata(belief)
Distributions.var(belief::Belief{T}) where { T <: Real }  = zero(T)
Distributions.std(belief::Belief{T}) where { T <: Real }  = zero(T)

as_belief(data)                     = Belief(data)
as_belief(belief::AbstractBelief)   = belief
as_belief(message::AbstractMessage) = Belief(getdata(message))
