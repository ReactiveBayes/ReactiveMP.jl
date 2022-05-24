export Dirichlet

import Distributions: Dirichlet
import SpecialFunctions: digamma

vague(::Type{<:Dirichlet}, dims::Int) = Dirichlet(ones(dims))

prod_analytical_rule(::Type{<:Dirichlet}, ::Type{<:Dirichlet}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Dirichlet, right::Dirichlet)
    mvec = probvec(left) .+ probvec(right)
    mvec = mvec .- one(eltype(mvec))
    return Dirichlet(mvec)
end

probvec(dist::Dirichlet) = params(dist)[1]

# probvec is not normalised
mean(::typeof(log), dist::Dirichlet) = digamma.(probvec(dist)) .- digamma(sum(probvec(dist)))

# Variate forms promotion

promote_variate_type(::Type{Multivariate}, ::Type{<:Dirichlet})  = Dirichlet
promote_variate_type(::Type{Matrixvariate}, ::Type{<:Dirichlet}) = MatrixDirichlet

promote_variate_type(::Type{Multivariate}, ::Type{<:MatrixDirichlet})  = Dirichlet
promote_variate_type(::Type{Matrixvariate}, ::Type{<:MatrixDirichlet}) = MatrixDirichlet
