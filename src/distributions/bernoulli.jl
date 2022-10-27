export Bernoulli, BernoulliNaturalParameters

import Distributions: Bernoulli, Distribution, succprob, failprob, logpdf
import Base
import StatsFuns: logistic

vague(::Type{<:Bernoulli}) = Bernoulli(0.5)

probvec(dist::Bernoulli) = (succprob(dist), failprob(dist))

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
    @assert length(probvec(right)) === 2 "Improper Bernoulli x Categorical product"
    return prod(ProdPreserveType(Bernoulli), left, Bernoulli(first(probvec(right))))
end

struct BernoulliNaturalParameters{T <: Real} <: NaturalParameters
    η::T
end

function Base.vec(p::BernoulliNaturalParameters)
    return [p.η]
end

function BernoulliNaturalParameters(v)
    return BernoulliNaturalParameters(v[1])
end

get_natural_params(params::BernoulliNaturalParameters) = params.η

function Base.:+(left::BernoulliNaturalParameters, right::BernoulliNaturalParameters)
    return BernoulliNaturalParameters(get_natural_params(left) + get_natural_params(right))
end

function Base.:-(left::BernoulliNaturalParameters, right::BernoulliNaturalParameters)
    return BernoulliNaturalParameters(get_natural_params(left) - get_natural_params(right))
end

function lognormalizer(η::BernoulliNaturalParameters)
    return log(logistic(-get_natural_params(η)))
end

function Distributions.logpdf(η::BernoulliNaturalParameters, x)
    return x * get_natural_params(η) + lognormalizer(η)
end

function convert(::Type{<:Distribution}, η::BernoulliNaturalParameters)
    return Bernoulli(exp(get_natural_params(η)) / (1 + exp(get_natural_params(η))))
end

function naturalparams(dist::Bernoulli)
    if succprob(dist) ≈ 1
        error("Bernoulli natural parameter is not defiend for p = 1.")
    end
    return BernoulliNaturalParameters(log(succprob(dist) / (1 - succprob(dist))))
end

isproper(params::BernoulliNaturalParameters) = true
