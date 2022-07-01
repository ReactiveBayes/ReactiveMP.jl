export InvWishart

import Distributions: InverseWishart, Wishart
import Base: ndims
import LinearAlgebra
import SpecialFunctions: digamma

struct InvWishart{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    ν::T
    S::A
end

InvWishart(ν::Real, S::AbstractMatrix{Real})  = InvWishart(promote(ν, S)...)
InvWishart(ν::Integer, S::AbstractMatrix{Real})  = InvWishart(float(ν), S)

Distributions.params(dist::InvWishart) = (dist.ν, dist.S)

Distributions.mean(dist::InvWishart)  = mean(InverseWishart(params(dist)...))
Distributions.var(dist::InvWishart)   = var(InverseWishart(params(dist)...))
Distributions.cov(dist::InvWishart)   = cov(InverseWishart(params(dist)...))
Distributions.mode(dist::InvWishart)  = mode(InverseWishart(params(dist)...))


Distributions.entropy(dist::InvWishart) = -Distributions.entropy(Wishart(dist.ν, inv(dist.S))) # not true

function Distributions.mean(::typeof(logdet), distribution::InverseWishart)
    d    = ndims(distribution)
    ν, S = params(distribution)
    return mapreduce(i -> -digamma((ν + 1 - i) / 2), +, 1:d) - d * log(2) - logdet(S)
end

vague(::Type{<:InvWishart}, dims::Int) = InverseWishart(dims, inv(Matrix(Diagonal(huge .* ones(dims)))))

# Base.ndims(dist::InverseWishart) = Distributions.dim(dist)

prod_analytical_rule(::Type{<:Wishart}, ::Type{<:Wishart}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::InvWishart, right::InvWishart)
    d = dim(left)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V  = lS + rS |> Hermitian |> Matrix
    df = ldf + rdf + d + 1

    return InverseWishart(df, V)
end
