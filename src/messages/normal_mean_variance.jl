
function multiply_messages(m1::Message{ <: NormalMeanVariance{T} }, m2::Message{ <: NormalMeanVariance{T} }) where { T <: Real }
    m1data = getdata(m1)
    m2data = getdata(m2)

    mean1 = mean(m1data)
    mean2 = mean(m2data)

    var1 = var(m1data)
    var2 = var(m2data)

    result = NormalMeanVariance((mean1 * var2 + mean2 * var1) / (var2 + var1), (var1 * var2) / (var1 + var2))

    return Message(result)
end