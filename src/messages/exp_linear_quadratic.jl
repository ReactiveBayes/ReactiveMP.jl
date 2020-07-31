
using Distributions

function multiply_messages(m1::Message{<:NormalMeanVariance{T}}, m2::Message{<:ExponentialLinearQuadratic{T}}) where T
    cubature = ghcubature(10)
    mean, var = approximate_meancov(ghcubature(10), (z) -> pdf(getdata(m2), z), getdata(m1))
    return NormalMeanVariance(mean, var)
end