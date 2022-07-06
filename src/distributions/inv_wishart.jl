export InvWishart

import Distributions: InverseWishart, Wishart
import Base: ndims
import LinearAlgebra
import StatsFuns: logπ
import SpecialFunctions: digamma, loggamma

struct InvWishart{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    ν::T
    S::A
end

function InvWishart(ν::Real, S::AbstractMatrix{<:Real})
    T = promote_type(typeof(ν), eltype(S))
    return InvWishart(convert(T, ν), convert(AbstractArray{T}, S))
end

InvWishart(ν::Integer, S::AbstractMatrix{Real}) = InvWishart(float(ν), S)

Distributions.params(dist::InvWishart) = (dist.ν, dist.S)

Distributions.mean(dist::InvWishart) = mean(InverseWishart(params(dist)...))
Distributions.var(dist::InvWishart)  = var(InverseWishart(params(dist)...))
Distributions.cov(dist::InvWishart)  = cov(InverseWishart(params(dist)...))
Distributions.mode(dist::InvWishart) = mode(InverseWishart(params(dist)...))
Distributions.dim(dist::InvWishart)  = size(dist.S, 1)

# from "Parametric Bayesian Estimation of Differential Entropy and Relative Entropy" Gupta et al.
function Distributions.entropy(dist::InvWishart)
    d = dim(dist)
    ν, S = params(dist)
    0.25 * d * (d - 1) * logπ + mapreduce(i -> loggamma(0.5 * (ν + 1.0 - i)), +, 1:d) +
    0.5ν * d + 0.5(d + 1) * (logdet(S) - log(2)) - 0.5(ν + d + 1) * sum([digamma(0.5(ν - d + i)) for i in 1:d])
    return 0
end

function Distributions.mean(::typeof(logdet), dist::InvWishart)
    d = dim(dist)
    ν, S = params(dist)
    return -(mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(2) - logdet(S))
end

function Distributions.mean(::typeof(inv), dist::InvWishart)
    ν, S = params(dist)
    return ν * cholinv(S)
end

vague(::Type{<:InvWishart}, dims::Integer) = InvWishart(dims, tiny .* diageye(dims))

function Base.convert(::Type{InverseWishart}, dist::InvWishart)
    ν, S = params(dist)
    return InverseWishart(ν, S)
end

function Base.convert(::Type{Wishart}, dist::InvWishart)
    ν, S = params(dist)
    return InverseWishart(ν, cholinv(S))
end

prod_analytical_rule(::Type{<:InvWishart}, ::Type{<:InvWishart}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::InvWishart, right::InvWishart)
    d = dim(left)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V = lS + rS |> Hermitian |> Matrix

    df = ldf + rdf + d + 1

    return InvWishart(df, V)
end
