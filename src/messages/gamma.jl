
import Distributions: Gamma

function multiply_messages(m1::Message{D}, m2::Message{D}) where { T, D <: Gamma{T} }
    return Message(Gamma(shape(m1) + shape(m2) - one(T), scale(m1) + scale(m2)))
end
