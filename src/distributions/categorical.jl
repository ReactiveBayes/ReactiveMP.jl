export Bernoulli

import Distributions: Categorical, probs

vague(::Type{ <: Categorical }, dims::Int) = Categorical(ones(dims) ./ dims)

prod_analytical_rule(::Type{ <: Categorical }, ::Type{ <: Categorical }) = ProdAnalyticalRuleAvailable()

convert_eltype(::Type{ Categorical }, ::Type{T}, distribution::Categorical{R}) where { T <: Real, R <: Real } = Categorical(convert(AbstractVector{T}, probs(distribution)))

function prod(::ProdAnalytical, left::Categorical, right::Categorical)
    # Multiplication of 2 categorical PMFs: p(z) = p(x) * p(y)
    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    norm = sum(mvec)
    return Categorical(mvec ./ norm)
end

probvec(dist::Categorical) = probs(dist)