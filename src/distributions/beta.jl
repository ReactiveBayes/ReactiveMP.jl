export Beta
export BetaNaturalParameters

import Distributions: Beta, params
import SpecialFunctions: digamma, logbeta, loggamma
import StatsFuns: betalogpdf

vague(::Type{<:Beta}) = Beta(1.0, 1.0)

prod_analytical_rule(::Type{<:Beta}, ::Type{<:Beta}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Beta, right::Beta)
    left_a, left_b   = params(left)
    right_a, right_b = params(right)
    T                = promote_samplefloattype(left, right)
    return Beta(left_a + right_a - one(T), left_b + right_b - one(T))
end

function compute_logscale(new_dist::Beta, left_dist::Beta, right_dist::Beta)
    return logbeta(params(new_dist)...) - logbeta(params(left_dist)...) - logbeta(params(right_dist)...)
end

function mean(::typeof(log), dist::Beta)
    a, b = params(dist)
    return digamma(a) - digamma(a + b)
end

function mean(::typeof(mirrorlog), dist::Beta)
    a, b = params(dist)
    return digamma(b) - digamma(a + b)
end

struct BetaNaturalParameters{T <: Real} <: NaturalParameters
    αm1::T
    βm1::T
end

BetaNaturalParameters(αm1::Real, βm1::Real)       = BetaNaturalParameters(promote(αm1, βm1)...)
BetaNaturalParameters(αm1::Integer, βm1::Integer) = BetaNaturalParameters(float(αm1), float(βm1))

Base.convert(::Type{BetaNaturalParameters}, a::Real, b::Real) = convert(BetaNaturalParameters{promote_type(typeof(a), typeof(b))}, a, b)

Base.convert(::Type{BetaNaturalParameters{T}}, a::Real, b::Real) where {T} = BetaNaturalParameters(convert(T, a), convert(T, b))

Base.convert(::Type{BetaNaturalParameters}, vec::AbstractVector) = convert(BetaNaturalParameters{eltype(vec)}, vec)

Base.convert(::Type{BetaNaturalParameters{T}}, vec::AbstractVector) where {T} = BetaNaturalParameters(convert(AbstractVector{T}, vec))

function isproper(params::BetaNaturalParameters)
    return ((params.αm1 + 1) > 0) && ((params.βm1 + 1) > 0)
end

naturalparams(dist::Beta) = BetaNaturalParameters(dist.α - 1, dist.β - 1)

function Base.convert(::Type{Distribution}, η::BetaNaturalParameters)
    return Beta(η.αm1 + 1, η.βm1 + 1, check_args = false)
end

function Base.vec(p::BetaNaturalParameters)
    return [p.αm1, p.βm1]
end

ReactiveMP.as_naturalparams(::Type{T}, args...) where {T <: BetaNaturalParameters} = convert(BetaNaturalParameters, args...)

function BetaNaturalParameters(v::AbstractVector{T}) where {T <: Real}
    @assert length(v) === 2 "`BetaNaturalParameters` must accept a vector of length `2`."
    return BetaNaturalParameters(v[1], v[2])
end

lognormalizer(params::BetaNaturalParameters) = logbeta(params.αm1 + 1, params.βm1 + 1)
logpdf(params::BetaNaturalParameters, x) = betalogpdf(params.αm1 + 1, params.βm1 + 1, x)

function Base.:-(left::BetaNaturalParameters, right::BetaNaturalParameters)
    return BetaNaturalParameters(left.αm1 - right.αm1, left.βm1 - right.βm1)
end
