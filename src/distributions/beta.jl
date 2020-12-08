export Beta

import Distributions: Beta, params
import SpecialFunctions: digamma

vague(::Type{ <: Beta }) = Beta(1.0, 1.0)

function prod(::ProdPreserveParametrisation, left::Beta{T}, right::Beta{T}) where T
    left_a, left_b = params(left)
    right_a, right_b = params(right)
    return Beta(left_a + right_a - one(Float64), left_b + right_b - one(Float64))
end

function logmean(dist::Beta) 
    a, b = params(dist)
    return digamma(a) - digamma(a + b)
end

function mirroredlogmean(dist::Beta)
    a, b = params(dist)
    return digamma(b) - digamma(a + b)
end