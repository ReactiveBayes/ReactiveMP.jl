export Wishart

import Distributions: Wishart
import Base: ndims, size, convert
import LinearAlgebra
import SpecialFunctions: digamma

"""
    WishartMessage

Same as `Wishart` from `Distributions.jl`, but does not check input arguments and allows creating improper `Wishart` message.
For model creation use `Wishart` from `Distributions.jl`. Regular user should never interact with `WishartMessage`.
"""
struct WishartMessage{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    ν::T
    S::A
end

function WishartMessage(ν::Real, S::AbstractMatrix{<:Real})
    T = promote_type(typeof(ν), eltype(S))
    return WishartMessage(convert(T, ν), convert(AbstractArray{T}, S))
end

WishartMessage(ν::Integer, S::AbstractMatrix{Real}) = WishartMessage(float(ν), S)

Distributions.params(dist::WishartMessage) = (dist.ν, dist.S)

Distributions.mean(dist::WishartMessage) = mean(Wishart(params(dist)...))
Distributions.var(dist::WishartMessage)  = var(Wishart(params(dist)...))
Distributions.cov(dist::WishartMessage)  = cov(Wishart(params(dist)...))
Distributions.mode(dist::WishartMessage) = mode(Wishart(params(dist)...))

Base.size(dist::WishartMessage, dim::Int) = size(dist.S, dim)

const WishartDistributionsFamily{T} = Union{Wishart{T}, WishartMessage{T}}

to_marginal(dist::WishartMessage) = convert(Wishart, dist)

function Base.convert(::Type{WishartMessage{T}}, distribution::WishartMessage) where {T}
    (ν, S) = params(distribution)
    return WishartMessage(convert(T, ν), convert(AbstractMatrix{T}, S))
end

function Distributions.mean(::typeof(logdet), distribution::WishartDistributionsFamily)
    d    = size(distribution, 1)
    ν, S = params(distribution)
    return mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(2) + logdet(S)
end

function Distributions.mean(::typeof(inv), distribution::WishartDistributionsFamily)
    ν, S = params(distribution)
    return ν * cholinv(S)
end

vague(::Type{<:Wishart}, dims::Int) = Wishart(dims, huge .* diageye(dims))

Base.ndims(dist::Wishart) = size(dist, 1)

function Base.convert(::Type{Wishart}, dist::WishartMessage)
    (ν, S) = params(dist)
    return Wishart(ν, Matrix(Hermitian(S)))
end

Base.convert(::Type{WishartMessage}, dist::Wishart) = WishartMessage(params(dist)...)

# We do not define prod between `Wishart` from `Distributions.jl` for a reason
# We want to compute `prod` only for `WishartMessage` messages as they are significantly faster in creation
prod_analytical_rule(::Type{<:WishartMessage}, ::Type{<:WishartMessage}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::WishartMessage, right::WishartMessage)
    @assert size(left, 1) === size(right, 1) "Cannot compute a product of two Wishart distributions of different sizes"

    d = size(left, 1)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V  = lS * cholinv(lS + rS) * rS
    df = ldf + rdf - d - 1

    return WishartMessage(df, V)
end
