
function multiply_messages(m1::Message{D}, m2::Message{D}) where { T, D <: Gamma{T} }
    return as_message(prod(ProdPreserveParametrisation(), getdata(m1), getdata(m2)))
end
