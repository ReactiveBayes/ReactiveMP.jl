
function multiply_messages(m1::Message{T}, m2::Message{T}) where { T <: Real }
    if abs(data(m1) - data(m2)) < eps(T)
        return Message(data(m1))
    else
        return Message(zero(T))
    end
end
