
function multiply_messages(m1::Message{Nothing}, m2::Message{Nothing})
    return Message(nothing)
end

@symmetrical function multiply_messages(::Message{Nothing}, message)
    return Message(getdata(message))
end
