export InvWishart

import Distributions: InverseWishart, Wishart
import Base: ndims
import LinearAlgebra
import SpecialFunctions: digamma, loggamma

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
Distributions.dim(dist::InvWishart)   = size(dist.S, 1)

# TODO:
# from "Parametric Bayesian Estimation of Differential Entropy and Relative Entropy" Gupta et al.
function Distributions.entropy(dist::InvWishart)
    d = dim(dist)
    ν, S = params(dist)
    mapreduce(i -> loggamma(0.5 * (ν + 1.0 - i)), +, 1:d) + 0.5ν*d + 0.5(d+1)*log(det(S)/2) + 0.5(ν+d+1) + sum([digamma(0.5(ν-d+i)) for i in 1:d])
end

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
