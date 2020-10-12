import Distributions: Normal

function multiply_messages(m1::Message{ <: Normal }, m2::Message{ <: Normal }) 

    mean1 = mean(m1)
    mean2 = mean(m2)

    var1 = var(m1)
    var2 = var(m2)

    result = Normal((mean1 * var2 + mean2 * var1) / (var2 + var1), sqrt((var1 * var2) / (var1 + var2)))

    return Message(result)
end
