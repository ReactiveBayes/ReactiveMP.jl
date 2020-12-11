export Dirichlet

import Distributions: Dirichlet
import SpecialFunctions: digamma

function prod(::ProdPreserveParametrisation, left::Dirichlet, right::Dirichlet)
    mvec = probvec(left) .+ probvec(right)
    mvec = mvec .- one(eltype(mvec))
    return Dirichlet(mvec)
end

probvec(dist::Dirichlet) = params(dist)[1]

logmean(dist::Dirichlet) = digamma.(probvec(dist)) .- digamma(sum(probvec(dist)))

vague(::Type{ <: Dirichlet }, dims::Int) = Dirichlet(ones(dims))