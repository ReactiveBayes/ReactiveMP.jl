export Beta

import Distributions: Beta, params
import SpecialFunctions: digamma, logbeta

vague(::Type{<:Beta}) = Beta(1.0, 1.0)

prod_analytical_rule(::Type{<:Beta}, ::Type{<:Beta}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Beta, right::Beta)
    left_a, left_b   = params(left)
    right_a, right_b = params(right)
    T                = promote_samplefloattype(left, right)
    return Beta(left_a + right_a - one(T), left_b + right_b - one(T))
end

function compute_logscale(new_dist::Beta, left_dist::Beta, right_dist::Beta)
    return logbeta(params(new_dist)...) - logbeta(params(left_dist)...) - logbeta(params(right_dist)...)
end

function mean(::typeof(log), dist::Beta)
    a, b = params(dist)
    return digamma(a) - digamma(a + b)
end

function mean(::typeof(mirrorlog), dist::Beta)
    a, b = params(dist)
    return digamma(b) - digamma(a + b)
end
