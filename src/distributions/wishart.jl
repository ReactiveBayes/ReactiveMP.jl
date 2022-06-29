export Wishart

import Distributions: Wishart
import Base: ndims
import LinearAlgebra
import SpecialFunctions: digamma

function Distributions.mean(::typeof(logdet), distribution::Wishart)
    d    = ndims(distribution)
    ν, S = params(distribution)
    return mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(2) + logdet(S)
end

vague(::Type{<:Wishart}, dims::Int) = Wishart(dims, Matrix(Diagonal(huge .* ones(dims))))

Base.ndims(dist::Wishart) = Distributions.dim(dist)

prod_analytical_rule(::Type{<:Wishart}, ::Type{<:Wishart}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Wishart, right::Wishart)
    d = dim(left)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V  = (lS * cholinv(lS + rS) * rS) |> Hermitian |> Matrix
    df = ldf + rdf - d - 1

    return Wishart(df, V)
end
