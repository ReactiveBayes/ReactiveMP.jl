export Wishart

import Distributions: Wishart
import Base: ndims

vague(::Type{ <: Wishart }, dims::Int) = Wishart(dims, Matrix(Diagonal(huge .* ones(dims))))

Base.ndims(dist::Wishart) = first(size(dist))

prod_analytical_rule(::Type{ <: Wishart }, ::Type{ <: Wishart }) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Wishart, right::Wishart)
    d = dim(left)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V  = (lS * cholinv(lS + rS) * rS) |> Hermitian |> Matrix
    df = ldf + rdf - d - 1

    return Wishart(df, V)
end