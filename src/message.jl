export AbstractMessage, Message, multiply_messages, reduce_messages
export AbstractBelief, Belief
export getdata

import Base: *
using Distributions

## AbstractMessage

abstract type AbstractMessage end

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

Distributions.mean(message::Message{T}) where { T <: Real } = getdata(message)
Distributions.var(message::Message{T}) where { T <: Real }  = zero(T)

## AbstractBelief

abstract type AbstractBelief end

## Belief

struct Belief{D} <: AbstractBelief
    data :: D
end

getdata(belief::Belief) = belief.data

function reduce_message_to_belief(messages)
    return Belief(messages |> reduce_messages |> getdata)
end

Distributions.mean(belief::Belief) = Distributions.mean(getdata(belief))
Distributions.var(belief::Belief)  = Distributions.var(getdata(belief))

Distributions.mean(belief::Belief{T}) where { T <: Real } = getdata(belief)
Distributions.var(belief::Belief{T}) where { T <: Real }  = zero(T)
