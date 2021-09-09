export GaussLaguerreQuadrature

using DomainIntegrals
using FastGaussQuadrature

import Base: ==


struct GaussLaguerreQuadrature{ R <: DomainIntegrals.HalfLineRule, W <: AbstractVector } <: AbstractApproximationMethod
    rule :: R
    logw :: W
end

GaussLaguerreQuadrature(n::Int) = GaussLaguerreQuadrature(Float64, n)

function GaussLaguerreQuadrature(::Type{T}, n::Int) where T 
    x, w = FastGaussQuadrature.gausslaguerre(n, zero(T))
    logw = log.(w)
    return GaussLaguerreQuadrature(DomainIntegrals.HalfLineRule(x, w), logw)
end

getpoints(approximation::GaussLaguerreQuadrature)     = points(approximation.rule)
getweights(approximation::GaussLaguerreQuadrature)    = weights(approximation.rule)
getlogweights(approximation::GaussLaguerreQuadrature) = approximation.logw
getlength(approximation::GaussLaguerreQuadrature)     = length(getweights(approximation))

approximation_name(approximation::GaussLaguerreQuadrature)       = "GaussLaguerre($(getlength(approximation)))"
approximation_short_name(approximation::GaussLaguerreQuadrature) = "GL$(getlength(approximation))"

approximate(approximation::GaussLaguerreQuadrature, fn::Function) = integral(approximation.rule, fn)

"""
This function calculates the log of the Gauss-laguerre integral by making use of the log of the integrable function.
    ln ( ∫ exp(-x)f(x) dx ) 
    ≈ ln ( ∑ wi * f(xi) ) 
    = ln ( ∑ exp( ln(wi) + logf(xi) ) )
    = ln ( ∑ exp( yi ) )
    = max(yi) + ln ( ∑ exp( yi - max(yi) ) )
    where we make use of the numerically stable log-sum-exp trick: https://en.wikipedia.org/wiki/LogSumExp
"""
function log_approximate(approximation::GaussLaguerreQuadrature, fn::Function)
    # get weights and points
    p    = getlength(approximation)
    x    = getpoints(approximation)
    logw = getlogweights(approximation)
    T    = eltype(logw)

    # calculate the ln(wi) + logf(xi) terms
    logresult = Vector{T}(undef, p)
    for i = 1:p
        logresult[i] = logw[i] + fn(x[i])
    end

    # return log sum exp
    return logsumexp(logresult)
end

function Base.:(==)(left::GaussLaguerreQuadrature{R}, right::GaussLaguerreQuadrature{R}) where R 
    return getlength(left) == getlength(right)
end