export Bernoulli

import Distributions: Bernoulli, succprob, failprob

function prod(::ProdPreserveParametrisation, left::Bernoulli, right::Bernoulli)
    left_p  = succprob(left)
    right_p = succprob(right)

    pprod = left_p * right_p 
    norm  = pprod + (one(T) - left_p) * (one(T) - right_p)
    @assert norm > 0 "Product of $(left) and $(right) results in non-normalizable distribution"
    return Bernoulli(pprod / norm)
end

probvec(dist::Bernoulli) = (succprob(dist), failprob(dist))