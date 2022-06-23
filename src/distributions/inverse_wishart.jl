export InverseWishart

import Distributions: InverseWishart
import Base: ndims
import LinearAlgebra
import SpecialFunctions: digamma

function Distributions.mean(::typeof(logdet), distribution::InverseWishart)
    d    = ndims(distribution)
    ν, S = params(distribution)
    invS = cholinv(S)
    return mapreduce(i -> -digamma((ν + 1 - i) / 2), +, 1:d) - d * log(2) - logdet(invS)
end

vague(::Type{<:InverseWishart}, dims::Int) = InverseWishart(dims, inv(Matrix(Diagonal(huge .* ones(dims)))))

Base.ndims(dist::InverseWishart) = Distributions.dim(dist)

prod_analytical_rule(::Type{<:Wishart}, ::Type{<:Wishart}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::InverseWishart, right::InverseWishart)
    d = dim(left)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V  = lS + rS |> Hermitian |> Matrix
    df = ldf + rdf + d + 1

    return InverseWishart(df, V)
end
