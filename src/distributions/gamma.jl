export Gamma

import Distributions: Gamma, shape, scale

vague(::Type{ <: Gamma }) = Gamma(1.0, huge)

function prod(::ProdPreserveParametrisation, left::Gamma{T}, right::Gamma{T}) where T
    return Gamma(shape(left) + shape(right) - one(T), (scale(left) * scale(right)) / (scale(left) + scale(right)))
end

function logmean(dist::Gamma)
    k, θ = params(dist)
    return digamma(k) + log(θ)
end

Base.convert(::Type{<:Gamma}, dist::GammaShapeRate) = Gamma(shape(dist), scale(dist))
Base.convert(::Type{<:GammaShapeRate}, dist::Gamma) = GammaShapeRate(shape(dist), rate(dist))

function prod(::ProdPreserveParametrisation, left::GammaShapeRate, right::Gamma)
    return GammaShapeRate(shape(left) + shape(right) - 1.0, rate(left) + rate(right))
end

function prod(::ProdPreserveParametrisation, left::Gamma, right::GammaShapeRate)
    return convert(Gamma, prod(ProdPreserveParametrisation(), right, left))
end
