export InverseWishart

import Distributions: InverseWishart, Wishart
import Base: ndims, size, convert
import LinearAlgebra
import StatsFuns: logπ
import SpecialFunctions: digamma, loggamma

"""
    InverseWishartMessage

Same as `InverseWishart` from `Distributions.jl`, but does not check input arguments and allows creating improper `InverseWishart` message.
For model creation use `InverseWishart` from `Distributions.jl`. Regular user should never interact with `InverseWishartMessage`.
"""
struct InverseWishartMessage{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    ν::T
    S::A
end

function InverseWishartMessage(ν::Real, S::AbstractMatrix{<:Real})
    T = promote_type(typeof(ν), eltype(S))
    return InverseWishartMessage(convert(T, ν), convert(AbstractArray{T}, S))
end

InverseWishartMessage(ν::Integer, S::AbstractMatrix{Real}) = InverseWishartMessage(float(ν), S)

Distributions.params(dist::InverseWishartMessage) = (dist.ν, dist.S)

Distributions.mean(dist::InverseWishartMessage) = mean(InverseWishart(params(dist)...))
Distributions.var(dist::InverseWishartMessage)  = var(InverseWishart(params(dist)...))
Distributions.cov(dist::InverseWishartMessage)  = cov(InverseWishart(params(dist)...))
Distributions.mode(dist::InverseWishartMessage) = mode(InverseWishart(params(dist)...))

Base.size(dist::InverseWishartMessage, dim::Int) = size(dist.S, dim)

const InverseWishartDistributionsFamily{T} = Union{InverseWishart{T}, InverseWishartMessage{T}}

to_marginal(dist::InverseWishartMessage) = convert(InverseWishart, dist)

function Base.convert(::Type{InverseWishartMessage{T}}, distribution::InverseWishartMessage) where {T}
    (ν, S) = params(distribution)
    return InverseWishartMessage(convert(T, ν), convert(AbstractMatrix{T}, S))
end

# from "Parametric Bayesian Estimation of Differential Entropy and Relative Entropy" Gupta et al.
function Distributions.entropy(dist::InverseWishartDistributionsFamily)
    d = size(dist, 1)
    ν, S = params(dist)
    d * (d - 1) / 4 * logπ + mapreduce(i -> loggamma((ν + 1.0 - i) / 2), +, 1:d) +
    ν / 2 * d + (d + 1) / 2 * (logdet(S) - log(2)) - (ν + d + 1) / 2 * mapreduce(i -> digamma((ν - d + i) / 2), +, 1:d)
end

function Distributions.mean(::typeof(logdet), dist::InverseWishartDistributionsFamily)
    d = size(dist, 1)
    ν, S = params(dist)
    return -(mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(2) - logdet(S))
end

function Distributions.mean(::typeof(inv), dist::InverseWishartDistributionsFamily)
    ν, S = params(dist)
    return ν * cholinv(S)
end

vague(::Type{<:InverseWishart}, dims::Integer) = InverseWishart(dims, tiny .* diageye(dims))

Base.ndims(dist::InverseWishart) = size(dist, 1)

function Base.convert(::Type{InverseWishart}, dist::InverseWishartMessage) 
    (ν, S) = params(dist)
    return InverseWishart(ν, Matrix(Hermitian(S)))
end

Base.convert(::Type{InverseWishartMessage}, dist::InverseWishart) = InverseWishartMessage(params(dist)...)

# We do not define prod between `InverseWishart` from `Distributions.jl` for a reason
# We want to compute `prod` only for `InverseWishartMessage` messages as they are significantly faster in creation
prod_analytical_rule(::Type{<:InverseWishartMessage}, ::Type{<:InverseWishartMessage}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::InverseWishartMessage, right::InverseWishartMessage)
    @assert size(left, 1) === size(right, 1) "Cannot compute a product of two InverseWishart distributions of different sizes"

    d = size(left, 1)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V = lS + rS

    df = ldf + rdf + d + 1

    return InverseWishartMessage(df, V)
end
