export ComplexNormal

import StatsFuns: logπ

struct ComplexNormal{T <: Complex} <: ContinuousUnivariateDistribution
    μ :: T
    Γ :: T
    C :: T
end

ComplexNormal(μ::Complex, Γ::Complex, C::Complex)   = ComplexNormal(promote(μ, Γ, C)...)
ComplexNormal(μ::Real, Γ::Real, C::Real)            = ComplexNormal(complex(μ), complex(Γ), complex(C))
ComplexNormal(μ::Real, Γ::Real)         	        = ComplexNormal(μ, Γ, zero(μ))
ComplexNormal(μ::Real)                              = ComplexNormal(μ, one(μ), zero(μ))
ComplexNormal(μ::Complex, Γ::Complex)         	    = ComplexNormal(μ, Γ, zero(μ))
ComplexNormal(μ::Complex, Γ::Real)         	        = ComplexNormal(μ, complex(Γ), zero(μ))
ComplexNormal(μ::Complex)                           = ComplexNormal(μ, one(μ), zero(μ))
ComplexNormal()                                     = ComplexNormal(0.0+0.0im, 1.0+0.0im, 0.0+0.0im)

# Distributions.@distr_support NormalMeanPrecision -Inf Inf

weightedmean(dist::ComplexNormal) = precision(dist) * mean(dist)

Distributions.mean(dist::ComplexNormal)    = dist.μ
Distributions.median(dist::ComplexNormal)  = mean(dist)
Distributions.mode(dist::ComplexNormal)    = mean(dist)
Distributions.var(dist::ComplexNormal)     = inv(dist.Γ)
Distributions.std(dist::ComplexNormal)     = sqrt(var(dist))
Distributions.cov(dist::ComplexNormal)     = var(dist)
Distributions.invcov(dist::ComplexNormal)  = dist.Γ
Distributions.entropy(dist::ComplexNormal) = 1 + logπ - log(real(precision(dist)))

# Distributions.pdf(dist::ComplexNormal, x::Real)    = (invsqrt2π * exp(-abs2(x - mean(dist)) * precision(dist) / 2)) * sqrt(precision(dist))
# Distributions.logpdf(dist::ComplexNormal, x::Real) = -(log2π - log(precision(dist)) + abs2(x - mean(dist)) * precision(dist)) / 2

Base.precision(dist::ComplexNormal)     = invcov(dist)
Base.eltype(::ComplexNormal{T}) where T = T

# Base.convert(::Type{ ComplexNormal }, μ::Real, w::Real) = ComplexNormal(μ, w)
# Base.convert(::Type{ ComplexNormal{T} }, μ::Real, w::Real) where { T <: Real } = NormalMeanPrecision(convert(T, μ), convert(T, w))

vague(::Type{ <: ComplexNormal }) = ComplexNormal(0.0, tiny, 0.0)

function Base.prod(::ProdPreserveParametrisation, left::ComplexNormal, right::ComplexNormal) 
    w = real(precision(left) + precision(right))
    μ = (mean(left) * real(precision(left)) + mean(right) * real(precision(right))) / w
    return ComplexNormal(μ, w, 0)
end
