export Beta

import Distributions: Beta, params
import SpecialFunctions: digamma

vague(::Type{ <: Beta }) = Beta(1.0, 1.0)

prod_analytical_rule(::Type{ <: Beta }, ::Type{ <: Beta }) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Beta, right::Beta)
    left_a, left_b   = params(left)
    right_a, right_b = params(right)
    T                = promote_type(eltype(left), eltype(right))
    return Beta(left_a + right_a - one(T), left_b + right_b - one(T))
end

function mean(::typeof(log), dist::Beta) 
    a, b = params(dist)
    return digamma(a) - digamma(a + b)
end

function mean(::typeof(mirrorlog), dist::Beta)
    a, b = params(dist)
    return digamma(b) - digamma(a + b)
end