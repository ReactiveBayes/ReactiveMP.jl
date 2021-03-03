export MatrixDirichlet

import SpecialFunctions: digamma, loggamma

struct MatrixDirichlet{T <: Real, A <: AbstractMatrix{T} } <: ContinuousMatrixDistribution
    a :: A
end

Distributions.mean(dist::MatrixDirichlet) = dist.a ./ sum(dist.a, dims = 1)

vague(::Type{ <: MatrixDirichlet }, dims::Vararg{Int}) = MatrixDirichlet(ones(dims...))
vague(::Type{ <: MatrixDirichlet }, dims::Tuple)       = MatrixDirichlet(ones(dims))

function Distributions.entropy(dist::MatrixDirichlet)
    return mapreduce(+, eachcol(dist.a)) do column
        scolumn = sum(column)
        -sum((column .- 1.0) .* (digamma.(column) .- digamma.(scolumn))) - loggamma(scolumn) + sum(loggamma.(column))
    end
end

logmean(dist::MatrixDirichlet) = digamma.(dist.a) .- digamma.(sum(dist.a, dims = 1))

function prod(::ProdPreserveParametrisation, left::MatrixDirichlet, right::MatrixDirichlet)
    return MatrixDirichlet(left.a + right.a .- 1.0)
end
