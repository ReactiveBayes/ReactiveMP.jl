export Message, data, multiply_messages

import Base: *

struct Message{D}
    data :: D
end

data(message::Message) = message.data

function multiply_messages end

Base.:*(m1::Message, m2::Message) = multiply_messages(m1, m2)
