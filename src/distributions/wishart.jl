export Wishart

import Distributions: Wishart

function prod(::ProdPreserveParametrisation, left::Wishart{T}, right::Wishart{T}) where T
    d = dim(left)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V  = (lS * cholinv(lS + rS) * rS) |> Hermitian |> Matrix
    df = ldf + rdf - d - one(ldf)

    return Wishart(df, V)
end