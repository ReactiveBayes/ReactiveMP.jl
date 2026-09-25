"""
    GammaShapeLikelihood(p, γ)

The likelihood of a Gamma distribution's shape `α`, `exp(γα - p⋅loggamma(α))` on the positive
half line, as the message towards `α` of a `GammaShapeRate` node. Two of them multiply by
adding their parameters.
"""
struct GammaShapeLikelihood{T <: Real} <: ContinuousUnivariateDistribution
    p::T
    γ::T # p * β
end

GammaShapeLikelihood(p::Real, γ::Real) = GammaShapeLikelihood(promote(p, γ)...)

Distributions.params(distribution::GammaShapeLikelihood) = (distribution.p, distribution.γ)

Distributions.@distr_support GammaShapeLikelihood 0.0 Inf

BayesBase.support(distribution::GammaShapeLikelihood{T}) where {T} = Distributions.RealInterval(zero(T), typemax(T))

BayesBase.logpdf(distribution::GammaShapeLikelihood, x::Real) = distribution.γ * x - distribution.p * loggamma(x)

BayesBase.default_prod_rule(::Type{<:GammaShapeLikelihood}, ::Type{<:GammaShapeLikelihood}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::GammaShapeLikelihood{T1}, right::GammaShapeLikelihood{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    return GammaShapeLikelihood(T(left.p) + T(right.p), T(left.γ) + T(right.γ))
end

BayesBase.paramfloattype(distribution::GammaShapeLikelihood) = BayesBase.deep_eltype(params(distribution))
BayesBase.convert_paramfloattype(::Type{T}, distribution::GammaShapeLikelihood) where {T} = GammaShapeLikelihood(T(distribution.p), T(distribution.γ))
