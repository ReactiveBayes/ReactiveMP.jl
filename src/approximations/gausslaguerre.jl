export GaussLaguerreQuadrature

using DomainIntegrals
using FastGaussQuadrature

import Base: ==


struct GaussLaguerreQuadrature{ R <: DomainIntegrals.HalfLineRule } <: AbstractApproximationMethod
    rule :: R
end

GaussLaguerreQuadrature(::Type{T}, n::Int) where T = GaussLaguerreQuadrature(DomainIntegrals.HalfLineRule(FastGaussQuadrature.gausslaguerre(n, zero(T))...))
GaussLaguerreQuadrature(n::Int)                    = GaussLaguerreQuadrature(DomainIntegrals.HalfLineRule(FastGaussQuadrature.gausslaguerre(n)...))

approximation_name(approx::GaussLaguerreQuadrature)       = "GaussLaguerre($(approx.rule))"
approximation_short_name(approx::GaussLaguerreQuadrature) = "GL$(approx.rule)"

approximate(approximation::GaussLaguerreQuadrature, fn::Function) = integral(approximation.rule, fn)

function Base.:(==)(left::GaussLaguerreQuadrature{R}, right::GaussLaguerreQuadrature{R}) where R 
    return length(weights(left.rule)) == length(weights(right.rule))
end