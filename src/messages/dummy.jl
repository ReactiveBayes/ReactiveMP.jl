
function multiply_messages(m1::Message{Nothing}, m2::Message{Nothing})
    # error("multiplyMessage(m1::Message{Nothing}, m2::Message{Nothing})")
    return Message(nothing)
end

function multiply_messages(m1::Message{Nothing}, m2)
    return Message(getdata(m2))
end

function multiply_messages(m1, m2::Message{Nothing})
    return Message(getdata(m1))
end
