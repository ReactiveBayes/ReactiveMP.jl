export GammaInverse
import Distributions: InverseGamma, shape, scale
import SpecialFunctions: digamma

const GammaInverse = InverseGamma

# uninformative prior
vague(::Type{<:GammaInverse}) = InverseGamma(2.0, huge)

convert_paramfloattype(::Type{T}, distribution::GammaInverse) where {T} = GammaInverse(convert.(T, params(distribution))...; check_args = false)

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
