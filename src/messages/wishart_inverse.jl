import Distributions: InverseWishart

function multiply_messages(m1::Message{ <: InverseWishart }, m2::Message{ <: InverseWishart }) 
    @assert size(m1) == size(m2)

    ν = m1.v + m.v - size(m1)[1] - one(eltype(m1))
    Ψ = inv(m1.Ψ) * (m1.Ψ + m2.Ψ) * inv(m2.Ψ)

    return Message(InverseWishart(ν, Ψ))
end
