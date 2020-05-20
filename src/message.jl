export Message, data
export multiplyMessages

using Distributions

import Base: *

struct Message{D}
    data :: D
end

data(message::Message) = message.data

function multiplyMessages(m1, m2) end

function multiplyMessages(m1::Message{Nothing}, m2::Message{Nothing})
    error("multiplyMessage(m1::Message{Nothing}, m2::Message{Nothing})")
end

function multiplyMessages(m1::Message{Nothing}, m2)
    return Message(data(m2))
end

function multiplyMessages(m1, m2::Message{Nothing})
    return Message(data(m1))
end

function multiplyMessages(m1::Message{T}, m2::Message{T}) where { T <: Real }
    if abs(data(m1) - data(m2)) < eps(T)
        return Message(data(m1))
    else
        return Message(zero(T))
    end
end

function multiplyMessages(m1::Message{N}, m2::Message{N}) where { N <: Normal }
    mean1 = mean(data(m1))
    mean2 = mean(data(m2))

    var1 = var(data(m1))
    var2 = var(data(m2))

    result = N((mean1 * var2 + mean2 * var1) / (var2 + var1), sqrt((var1 * var2) / (var1 + var2)))

    return Message(result)
end

Base.:*(m1::Message, m2::Message) = multiplyMessages(m1, m2)
