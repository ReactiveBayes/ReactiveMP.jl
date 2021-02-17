export GaussLaguerreQuadrature

using DomainIntegrals

import Base: ==


struct GaussLaguerreQuadrature{ R <: DomainIntegrals.HalfLineRule } <: AbstractApproximationMethod
    rule :: R
end

GaussLaguerreQuadrature(n::Int) = GaussLaguerreQuadrature(Q_GaussLaguerre(n))

approximation_name(approx::GaussLaguerreQuadrature)       = "GaussLaguerre($(approx.rule))"
approximation_short_name(approx::GaussLaguerreQuadrature) = "GL$(approx.rule)"

approximate(approximation::GaussLaguerreQuadrature, fn::Function) = integral(approximation.rule, fn)

function Base.:(==)(left::GaussLaguerreQuadrature{R}, right::GaussLaguerreQuadrature{R}) where R 
    return length(weights(left.rule)) == length(weights(right.rule))
end