import Distributions: Wishart

function multiply_messages(m1::Message{ <: Wishart }, m2::Message{ <: Wishart }) 
    @assert size(m1) == size(m2)

    ν = m1.v + m.v - size(m1)[1] - one(eltype(m1))
    S = m1.S * inv(m1.S + m2.S) * m2.S

    return Message(Wishart(ν, S))
end
