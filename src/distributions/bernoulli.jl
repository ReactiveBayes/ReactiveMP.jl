export Bernoulli
export naturalParams
export standardDist

import Distributions: Bernoulli, succprob, failprob
import Base

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

struct BernoulliNaturalParameters{T} <: NaturalParameters
    η::T
end

get_natural_params(params::BernoulliNaturalParameters) = params.η

function Base.:+(left::BernoulliNaturalParameters, right::BernoulliNaturalParameters)
    return BernoulliNaturalParameters(get_natural_params(left) + get_natural_params(right))
end

function Base.:-(left::BernoulliNaturalParameters, right::BernoulliNaturalParameters)
    return BernoulliNaturalParameters(get_natural_params(left) - get_natural_params(right))
end

function logNormalizer(η::BernoulliNaturalParameters)
    return log(1 + exp(get_natural_params(η)))
end

function log_normalizer(::Type{BernoulliNaturalParameters})
    return (v) -> log(1 + exp(v))
end

function logPdf(η::BernoulliNaturalParameters, x)
    return x * get_natural_params(η) - logNormalizer(η)
end

function standardDist(η::BernoulliNaturalParameters)
    return Bernoulli(get_natural_params(η) / (1 + exp(get_natural_params(η))))
end

naturalParams(dist::Bernoulli) = BernoulliNaturalParameters(log(dist.params[:p] / (1 - dist.params[:p])))
