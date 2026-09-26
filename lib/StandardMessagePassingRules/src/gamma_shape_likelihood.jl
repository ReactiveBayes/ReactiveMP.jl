"""
    GammaShapeLikelihood(p::Real, γ::Real)

The likelihood of a Gamma distribution's shape `α`, the unnormalised density

```math
f(α) = \\exp(γ α - p \\log Γ(α)), \\quad α > 0,
```

which is the message towards the shape of a `GammaShapeRate` node, with `p = 1` and
`γ = E[log β] + E[log out]`, and towards a shape `(:a, k)` of a [`GammaMixture`](@ref), with
`p` the component's responsibility. `p` and `γ` are promoted to one type.

It has `logpdf`, `support` (the positive half line) and `params`, `(p, γ)`, but no mean or
normaliser. The product of two of them is a `GammaShapeLikelihood` again, with the parameters
added; with any other distribution the default product is a lazy `ProductOf`.

# Examples

```jldoctest; setup = :(using StandardMessagePassingRules, BayesBase, Distributions)
julia> left, right = GammaShapeLikelihood(1.0, 2.0), GammaShapeLikelihood(0.5, -1.0);

julia> product = prod(GenericProd(), left, right);

julia> product isa GammaShapeLikelihood && all(params(product) .≈ (1.5, 1.0))
true

julia> logpdf(left, 1.0) ≈ 2.0
true
```
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
