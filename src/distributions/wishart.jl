export Wishart

import Distributions: Wishart
import Base: ndims, size, convert
import LinearAlgebra
import SpecialFunctions: digamma

"""
    WishartMessage

Same as `Wishart` from `Distributions.jl`, but does not check input arguments and allows creating improper `Wishart` message.
For model creation use `Wishart` from `Distributions.jl`. Regular user should never interact with `WishartMessage`.

Note that internally `WishartMessage` stores (and creates with) inverse of its-scale matrix, but (for backward compatibility) `params()` function returns the scale matrix itself. 
This is done for better stability in the message passing update rules.
"""
struct WishartMessage{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    ν    :: T
    invS :: A
end

function WishartMessage(ν::Real, invS::AbstractMatrix{<:Real})
    T = promote_type(typeof(ν), eltype(invS))
    return WishartMessage(convert(T, ν), convert(AbstractArray{T}, invS))
end

WishartMessage(ν::Integer, invS::AbstractMatrix{Real}) = WishartMessage(float(ν), invS)

Distributions.params(dist::WishartMessage)  = (dist.ν, cholinv(dist.invS))
Distributions.mean(dist::WishartMessage)    = mean(convert(Wishart, dist))
Distributions.var(dist::WishartMessage)     = var(convert(Wishart, dist))
Distributions.cov(dist::WishartMessage)     = cov(convert(Wishart, dist))
Distributions.mode(dist::WishartMessage)    = mode(convert(Wishart, dist))
Distributions.entropy(dist::WishartMessage) = entropy(convert(Wishart, dist))

mean_cov(dist::WishartMessage) = mean_cov(convert(Wishart, dist))

Base.size(dist::WishartMessage)           = size(dist.invS)
Base.size(dist::WishartMessage, dim::Int) = size(dist.invS, dim)

const WishartDistributionsFamily{T} = Union{Wishart{T}, WishartMessage{T}}

to_marginal(dist::WishartMessage) = convert(Wishart, dist)

function Base.convert(::Type{WishartMessage{T}}, distribution::WishartMessage) where {T}
    (ν, invS) = (distribution.ν, distribution.invS)
    return WishartMessage(convert(T, ν), convert(AbstractMatrix{T}, invS))
end

function Distributions.mean(::typeof(logdet), distribution::WishartMessage)
    d       = size(distribution, 1)
    ν, invS = (distribution.ν, distributions.invS)
    return mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(2) - logdet(invS)
end

function Distributions.mean(::typeof(logdet), distribution::Wishart)
    d = size(distribution, 1)
    ν, S = params(distribution)
    return mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(2) + logdet(S)
end

function Distributions.mean(::typeof(inv), distribution::WishartDistributionsFamily)
    return mean(cholinv, distribution)
end

function Distributions.mean(::typeof(cholinv), distribution::WishartMessage)
    ν, invS = (distribution.ν, distribution.invS)
    return mean(InverseWishart(ν, invS))
end

function Distributions.mean(::typeof(cholinv), distribution::Wishart)
    ν, S = params(distribution)
    return mean(InverseWishart(ν, cholinv(S)))
end

vague(::Type{<:Wishart}, dims::Int) = Wishart(dims, huge .* diageye(dims))

Base.ndims(dist::Wishart) = size(dist, 1)

function Base.convert(::Type{Wishart}, dist::WishartMessage)
    (ν, S) = params(dist)
    return Wishart(ν, Matrix(Hermitian(S)))
end

function Base.convert(::Type{WishartMessage}, dist::Wishart)
    (ν, S) = params(dist)
    return WishartMessage(ν, cholinv(S))
end

## Friendly functions

function logpdf_sample_friendly(dist::WishartMessage)
    friendly = convert(Wishart, dist)
    return (friendly, friendly)
end

# We do not define prod between `Wishart` from `Distributions.jl` for a reason
# We want to compute `prod` only for `WishartMessage` messages as they are significantly faster in creation
prod_analytical_rule(::Type{<:WishartMessage}, ::Type{<:WishartMessage}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::WishartMessage, right::WishartMessage)
    @assert size(left, 1) === size(right, 1) "Cannot compute a product of two Wishart distributions of different sizes"

    d = size(left, 1)

    ldf, linvS = (left.ν, left.invS)
    rdf, rinvS = (right.ν, right.invS)

    # See Matrix Cookbook 
    # 3.2.5 The Searle Set of Identities - eq (163)
    # V  = lS * cholinv(lS + rS) * rS
    invV = linvS + rinvS
    df   = ldf + rdf - d - 1

    return WishartMessage(df, invV)
end
