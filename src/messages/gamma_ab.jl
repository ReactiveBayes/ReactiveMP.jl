
function multiply_messages(m1::Message{D}, m2::Message{D}) where { T, D <: GammaAB{T} }
    d1 = getdata(m1)
    d2 = getdata(m2)
    return Message(GammaAB(d1.a + d2.a - one(T), d1.b + d2.b))
end
