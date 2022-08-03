export GammaInverse
import Distributions: InverseGamma, shape, scale

const GammaInverse = InverseGamma

# TODO: which functions return Distribution.InverseGamma and which return ReactiveMP.GammaInverse?
import SpecialFunctions: digamma

# uninformative prior
vague(::Type{<:GammaInverse}) = InverseGamma(1.0, tiny)

prod_analytical_rule(::Type{<:GammaInverse}, ::Type{<:GammaInverse}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::GammaInverse, right::InverseGamma)
    return GammaInverse(shape(left) + shape(right) + 1, scale(left) + scale(right))
end

function mean(::typeof(log), dist::GammaInverse)
    α = shape(dist)
    θ = scale(dist)
    return log(θ) - digamma(α)
end

function mean(::typeof(inv), dist::GammaInverse)
    α = shape(dist)
    θ = scale(dist)
    return α / θ
end
