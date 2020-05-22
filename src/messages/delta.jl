
function multiply_messages(m1::Message{T}, m2::Message{T}) where { T <: Real }
    if abs(getdata(m1) - getdata(m2)) < eps(T)
        return Message(getdata(m1))
    else
        return Message(zero(T))
    end
end
