
using Distributions

function multiply_messages(m1::Message{<:NormalMeanVariance{T}}, m2::Message{<:ExponentialLinearQuadratic{T}}) where T
    return as_message(prod(ProdPreserveParametrisation(), getdata(m1), getdata(m2)))
end