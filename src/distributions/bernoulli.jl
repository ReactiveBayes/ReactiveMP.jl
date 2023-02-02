export Bernoulli, BernoulliNaturalParameters

import Distributions: Bernoulli, Distribution, succprob, failprob, logpdf
import Base
import StatsFuns: logistic

vague(::Type{<:Bernoulli}) = Bernoulli(0.5)

probvec(dist::Bernoulli) = (failprob(dist), succprob(dist))

prod_analytical_rule(::Type{<:Bernoulli}, ::Type{<:Bernoulli}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Bernoulli, right::Bernoulli)
    left_p  = succprob(left)
    right_p = succprob(right)

    pprod = left_p * right_p
    norm  = pprod + (one(left_p) - left_p) * (one(right_p) - right_p)
    @assert norm > 0 "Product of $(left) and $(right) results in non-normalizable distribution"
    return Bernoulli(pprod / norm)
end

prod_analytical_rule(::Type{<:Bernoulli}, ::Type{<:Categorical}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Bernoulli, right::Categorical)

    # get probability vectors
    p_left = probvec(left)
    p_right = probvec(right)

    # find length of new vector and compute entries
    if length(p_left) >= length(p_right)
        p_new = vcat(p_right..., zeros(length(p_left) - length(p_right)))
        p_new .*= p_left
    else
        p_new = vcat(p_left..., zeros(length(p_right) - length(p_left)))
        p_new .*= p_right
    end

    # return categorical with normalized probability vector
    return Categorical(normalize!(p_new, 1))
end

function prod(::AddonProdLogScale, new_dist::Bernoulli, left_dist::Bernoulli, right_dist::Bernoulli)
    left_p = succprob(left_dist)
    right_p = succprob(right_dist)
    a = left_p * right_p + (one(left_p) - left_p) * (one(right_p) - right_p)
    return log(a)
end

function prod(::AddonProdLogScale, new_dist::Categorical, left_dist::Bernoulli, right_dist::Categorical)

    # get probability vectors
    p_left = probvec(left)
    p_right = probvec(right)

    # find length of new vector and compute entries
    if length(p_left) >= length(p_right)
        p_new = vcat(p_right..., zeros(length(p_left) - length(p_right)))
        Z = dot(p_left, p_new)
    else
        p_new = vcat(p_left..., zeros(length(p_right) - length(p_left)))
        Z = dot(p_right, p_new)
    end

    # return log normalization constant
    return log(Z)
end

struct BernoulliNaturalParameters{T <: Real} <: NaturalParameters
    η::T
end

function Base.vec(p::BernoulliNaturalParameters)
    return [p.η]
end

function BernoulliNaturalParameters(v::AbstractVector)
    @assert length(v) === 1 "`BernoulliNaturalParameters` must accept a vector of length `1`."
    return BernoulliNaturalParameters(v[1])
end

Base.convert(::Type{BernoulliNaturalParameters}, η::Real) = convert(BernoulliNaturalParameters{typeof(η)}, η)

Base.convert(::Type{BernoulliNaturalParameters{T}}, η::Real) where {T} = BernoulliNaturalParameters(convert(T, η))

Base.convert(::Type{BernoulliNaturalParameters}, vec::AbstractVector) = convert(BernoulliNaturalParameters{eltype(vec)}, vec)

Base.convert(::Type{BernoulliNaturalParameters{T}}, vec::AbstractVector) where {T} = BernoulliNaturalParameters(convert(AbstractVector{T}, vec))

function Base.:(==)(left::BernoulliNaturalParameters, right::BernoulliNaturalParameters)
    return left.η == right.η
end

as_naturalparams(::Type{T}, args...) where {T <: BernoulliNaturalParameters} = convert(BernoulliNaturalParameters, args...)

function Base.:+(left::BernoulliNaturalParameters, right::BernoulliNaturalParameters)
    return BernoulliNaturalParameters(left.η + right.η)
end

function Base.:-(left::BernoulliNaturalParameters, right::BernoulliNaturalParameters)
    return BernoulliNaturalParameters(left.η - right.η)
end

function lognormalizer(params::BernoulliNaturalParameters)
    return log(logistic(-params.η))
end

function Distributions.logpdf(params::BernoulliNaturalParameters, x)
    return x * params.η + lognormalizer(params)
end

function convert(::Type{<:Distribution}, params::BernoulliNaturalParameters)
    return Bernoulli(exp(params.η) / (1 + exp(params.η)))
end

function naturalparams(dist::Bernoulli)
    @assert !(succprob(dist) ≈ 1) "Bernoulli natural parameters are not defiend for p = 1."
    return BernoulliNaturalParameters(log(succprob(dist) / (1 - succprob(dist))))
end

isproper(params::BernoulliNaturalParameters) = true
