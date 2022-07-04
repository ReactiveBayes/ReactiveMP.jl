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

function Distributions.mean(::typeof(inv), distribution::Wishart)
    ν, S = params(distribution)
    return ν*inv(S)
end

vague(::Type{<:Wishart}, dims::Int) = Wishart(dims, Matrix(Diagonal(huge .* ones(dims))))

Base.ndims(dist::Wishart) = size(dist, 1)

prod_analytical_rule(::Type{<:Wishart}, ::Type{<:Wishart}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Wishart, right::Wishart)
    @assert size(left, 1) === size(right, 1) "Cannot compute a product of two Wishart distributions of different sizes"

    d = size(left, 1)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V  = (lS * cholinv(lS + rS) * rS) |> Hermitian |> Matrix
    df = ldf + rdf - d - 1

    return Wishart(df, V)
end
