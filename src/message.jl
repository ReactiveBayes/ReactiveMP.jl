export Message, multiply_messages, reduce_messages
export Belief
export getdata

import Base: *

struct Message{D}
    data :: D
end

getdata(message::Message) = message.data

function multiply_messages end

function reduce_messages(messages)
    return reduce(*, messages; init = nothing)
end

Base.:*(m1::Message, m2::Message) = multiply_messages(m1, m2)

struct Belief{D}
    data :: D
end

getdata(belief::Belief) = belief.data

function reduce_message_to_belief(messages)
    return Belief(messages |> reduce_messages |> getdata)
end
