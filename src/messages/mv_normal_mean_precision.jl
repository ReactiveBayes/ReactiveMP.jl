

function multiply_messages(m1::Message{ <: MvNormalMeanPrecision }, m2::Message{ <: MvNormalMeanPrecision })
    return as_message(prod(ProdPreserveParametrisation(), getdata(m1), getdata(m2)))
end