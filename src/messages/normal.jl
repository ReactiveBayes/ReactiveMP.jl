import Distributions: Normal

function multiply_messages(m1::Message{ <: Normal }, m2::Message{ <: Normal }) 
    m1data = getdata(m1)
    m2data = getdata(m2)

    mean1 = mean(m1data)
    mean2 = mean(m2data)

    var1 = var(m1data)
    var2 = var(m2data)

    result = Normal((mean1 * var2 + mean2 * var1) / (var2 + var1), sqrt((var1 * var2) / (var1 + var2)))

    return Message(result)
end
