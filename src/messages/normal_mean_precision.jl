
function multiply_messages(m1::Message{ <: NormalMeanPrecision{T} }, m2::Message{ <: NormalMeanPrecision{T} }) where { T }
    return as_message(prod(ProdPreserveParametrisation(), getdata(m1), getdata(m2)))
end
