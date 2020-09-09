using Distributions

function multiply_messages(m1::Message{ <: NormalMeanPrecision{T} }, m2::Message{ <: NormalMeanPrecision{T} }) where { T }
    mean1 = mean(m1)
    mean2 = mean(m2)

    var1 = var(m1)
    var2 = var(m2)

    result = NormalMeanPrecision((mean1 * var2 + mean2 * var1) / (var1 + var2), (var1 + var2) / (var1 * var2))
    return Message(result)
end

@symmetrical function multiply_messages(m1::Message{ <: Dirac{T} }, m2::Message{ <: NormalMeanPrecision{T} }) where { T <: Real }
    mean1 = mean(m1)
    mean2 = mean(m2)

    var1 = var(m1)
    var2 = var(m2)

    result = NormalMeanPrecision((mean1 * var2 + mean2 * var1) / (var1 + var2), (var1 + var2) / (var1 * var2))
    return Message(result)
end
