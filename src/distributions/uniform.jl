export Uniform

import Distributions: Uniform

vague(::Type{<:Uniform}) = Uniform(-100.0, 100.0)

convert_paramfloattype(::Type{T}, dist::Uniform) where{T} = Uniform(convert_paramfloattype.(T, params(dist))...)
