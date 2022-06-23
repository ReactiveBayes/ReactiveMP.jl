export InverseWishart

import Distributions: InverseWishart
import Base: ndims
import LinearAlgebra
import SpecialFunctions: digamma

function Distributions.mean(::typeof(logdet), distribution::InverseWishart)
    d    = ndims(distribution)
    wishart = convert(Wishart, dist)
    ν, S = params(wishart)
    return mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(2) + logdet(S)
end

vague(::Type{<:InverseWishart}, dims::Int) = InverseWishart(dims, inv(Matrix(Diagonal(huge .* ones(dims)))))

Base.ndims(dist::InverseWishart) = Distributions.dim(dist)

prod_analytical_rule(::Type{<:Wishart}, ::Type{<:Wishart}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::InverseWishart, right::InverseWishart)
    res = prod(ProdAnalytical(), convert(Wishart, left), convert(Wishart, right))
    ν, S = params(res)
    return InverseWishart(ν, inv(S))
end
