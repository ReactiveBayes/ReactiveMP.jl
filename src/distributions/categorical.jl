export Bernoulli

import Distributions: Categorical, probs

vague(::Type{ <: Categorical }, dims::Int) = Categorical(ones(dims) ./ dims)

function prod(::ProdPreserveParametrisation, left::Categorical, right::Categorical)
    # Multiplication of 2 categorical PMFs: p(z) = p(x) * p(y)
    mvec = clamp.(probvec(left) .* probvec(right), tiny, Inf)
    norm = sum(mvec)
    return Categorical(mvec ./ norm)
end

probvec(dist::Categorical) = probs(dist)