export Categorical

import Distributions: Categorical, probs

vague(::Type{<:Categorical}, dims::Int) = Categorical(ones(dims) ./ dims)

convert_paramfloattype(::Type{T}, distribution::Categorical) where {T <: Real} = Categorical(convert_paramfloattype(T, probs(distribution)))

prod_analytical_rule(::Type{<:Categorical}, ::Type{<:Categorical}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Categorical, right::Categorical)
    # Multiplication of 2 categorical PMFs: p(z) = p(x) * p(y)
    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    norm = sum(mvec)
    return Categorical(mvec ./ norm)
end

probvec(dist::Categorical) = probs(dist)

function compute_logscale(new_dist::Categorical, left_dist::Categorical, right_dist::Categorical)
    return log(dot(probvec(left_dist), probvec(right_dist)))
end
