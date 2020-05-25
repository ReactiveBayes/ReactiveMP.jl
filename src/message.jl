export AbstractMessage, Message, multiply_messages, reduce_messages
export AbstractBelief, Belief
export getdata

import Base: *

abstract type AbstractMessage end

struct Message{D} <: AbstractMessage
    data :: D
end

getdata(message::Message) = message.data

function multiply_messages end

function reduce_messages(messages)
    return reduce(*, messages; init = Message(nothing))
end

Base.:*(m1::AbstractMessage, m2::AbstractMessage) = multiply_messages(m1, m2)

abstract type AbstractBelief end

struct Belief{D} <: AbstractBelief
    data :: D
end

getdata(belief::Belief) = belief.data

function reduce_message_to_belief(messages)
    return Belief(messages |> reduce_messages |> getdata)
end
