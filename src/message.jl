export AbstractMessage, DeterministicMessage, StochasticMessage
export multiply

using Distributions
using Rocket

import Base: *

abstract type AbstractMessage end

struct DeterministicMessage <: AbstractMessage
    value :: Float64
end

struct StochasticMessage{D} <: AbstractMessage
    distribution :: D
end

function multiply(d1::DeterministicMessage, d2::DeterministicMessage)::DeterministicMessage
    if abs(d1.value - d2.value) < eps(Float64)
        return DeterministicMessage(d1.value)
    end
    return DeterministicMessage(zero(Float64))
end

function multiply(n1::StochasticMessage{Normal{Float64}}, n2::StochasticMessage{Normal{Float64}})::StochasticMessage{Normal{Float64}}
    mean1 = mean(n1.distribution)
    mean2 = mean(n2.distribution)

    var1 = var(n1.distribution)
    var2 = var(n2.distribution)

    result = Normal((mean1 * var2 + mean2 * var1) / (var2 + var1), sqrt((var1 * var2) / (var1 + var2)))

    return StochasticMessage(result)
end

multiply(m1::AbstractMessage, m2::AbstractMessage) = error("Message multiplication for types $(typeof(m1)) and $(typeof(m2)) is not implemented")

Base.:*(m1::AbstractMessage, m2::AbstractMessage) = multiply(m1, m2)
