export Bernoulli

import Distributions: Bernoulli, succprob, failprob

vague(::Type{ <: Bernoulli }) = Bernoulli(0.5)

prod_analytical_rule(::Type{ <: Bernoulli }, ::Type{ <: Bernoulli }) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Bernoulli, right::Bernoulli)
    left_p  = succprob(left)
    right_p = succprob(right)

    pprod = left_p * right_p 
    norm  = pprod + (one(left_p) - left_p) * (one(right_p) - right_p)
    @assert norm > 0 "Product of $(left) and $(right) results in non-normalizable distribution"
    return Bernoulli(pprod / norm)
end

probvec(dist::Bernoulli) = (succprob(dist), failprob(dist))