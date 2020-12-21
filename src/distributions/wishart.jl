export Wishart

import Distributions: Wishart
import Base: ndims

vague(::Type{ <: Wishart }, dims::Int) = Wishart(dims, Matrix(Diagonal(1.0e-20 .* ones(dims))))

Base.ndims(dist::Wishart) = size(dist)[1]

function prod(::ProdPreserveParametrisation, left::Wishart{T}, right::Wishart{T}) where T
    d = dim(left)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V  = (lS * cholinv(lS + rS) * rS) |> Hermitian |> Matrix
    df = ldf + rdf - d - one(ldf)

    return Wishart(df, V)
end