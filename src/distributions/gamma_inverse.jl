export GammaInverse
import Distributions: InverseGamma, shape, scale, cov

const GammaInverse = InverseGamma

# TODO: which functions return Distribution.InverseGamma and which return ReactiveMP.GammaInverse?
import Distributions: InverseGamma, shape, scale, cov
import SpecialFunctions: digamma

# TODO: ?
vague(::Type{<:InverseGamma}) = InverseGamma(1.0, 1.0)

prod_analytical_rule(::Type{<:InverseGamma}, ::Type{<:InverseGamma}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::InverseGamma, right::InverseGamma)
    return InverseGamma(shape(left) + shape(right) + 1, scale(left) + scale(right))
end

function mean(::typeof(log), dist::InverseGamma)
    α = scale(dist)
    β = shape(dist)
    return log(α) - digamma(β)
end

function mean(::typeof(inv), dist::InverseGamma)
    α = scale(dist)
    β = shape(dist)
    return β / α
end
