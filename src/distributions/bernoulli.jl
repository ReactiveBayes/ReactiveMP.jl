export Bernoulli

import Distributions: Bernoulli, succprob, failprob

vague(::Type{<:Bernoulli}) = Bernoulli(0.5)

probvec(dist::Bernoulli) = (succprob(dist), failprob(dist))

prod_analytical_rule(::Type{<:Bernoulli}, ::Type{<:Bernoulli}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Bernoulli, right::Bernoulli)
    left_p  = succprob(left)
    right_p = succprob(right)

    pprod = left_p * right_p
    norm  = pprod + (one(left_p) - left_p) * (one(right_p) - right_p)
    @assert norm > 0 "Product of $(left) and $(right) results in non-normalizable distribution"
    return Bernoulli(pprod / norm)
end

prod_analytical_rule(::Type{<:Bernoulli}, ::Type{<:Categorical}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Bernoulli, right::Categorical)
    @assert length(probvec(right)) === 2 "Improper Bernoulli x Categorical product"
    return prod(ProdPreserveType(Bernoulli), left, Bernoulli(first(probvec(right))))
end

function prod(::AddonProdLogScale, new_dist::Bernoulli, left_dist::Bernoulli, right_dist::Bernoulli)
    left_p = succprob(left_dist)
    right_p = succprob(right_dist)
    a = left_p * right_p + (one(left_p) - left_p) * (one(right_p) - right_p)
    return log(a)
end

function prod(::AddonProdLogScale, new_dist::Bernoulli, left_dist::Bernoulli, right_dist::Categorical)
    @assert length(probvec(right_dist)) === 2 "Improper Bernoulli x Categorical product"
    left_p = succprob(left_dist)
    right_p = first(probvec(right_dist))
    a = left_p * right_p + (one(left_p) - left_p) * (one(right_p) - right_p)
    return log(a)
end