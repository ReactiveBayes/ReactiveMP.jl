

function multiply_messages(m1::Message{N}, m2::Message{N}) where { N <: MvNormalMeanPrecision }
    return as_message(prod(ProdPreserveParametrisation(), getdata(m1), getdata(m2)))
end