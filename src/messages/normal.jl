import Distributions: Normal

function multiply_messages(m1::Message{N}, m2::Message{N}) where { N <: Normal }
    mean1 = mean(data(m1))
    mean2 = mean(data(m2))

    var1 = var(data(m1))
    var2 = var(data(m2))

    result = N((mean1 * var2 + mean2 * var1) / (var2 + var1), sqrt((var1 * var2) / (var1 + var2)))

    return Message(result)
end
