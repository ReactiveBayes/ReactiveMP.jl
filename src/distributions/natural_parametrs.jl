export NaturalParameters
import Base: convert

abstract type NaturalParameters end

Base.convert(::Type{T}, params::NaturalParameters) where {T <: Distribution} = convert(T, convert(Distribution, params))
