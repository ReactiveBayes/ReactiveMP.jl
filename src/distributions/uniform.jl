export Uniform

import Distributions: Uniform

vague(::Type{<:Uniform}) = Uniform(-100.0, 100.0)
