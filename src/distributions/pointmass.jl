export PointMass, getpointmass

import Distributions: mean, var, cov, std, insupport, pdf, logpdf, entropy
import Base: ndims, precision, getindex
import SpecialFunctions: loggamma, logbeta

struct PointMass{P}
    point :: P
end

as_marginal(distribution::PointMass)  = Marginal(distribution)
getpointmass(distribution::PointMass) = distribution.point

logmean(distribution::PointMass)      = log(mean(distribution))
inversemean(distribution::PointMass)  = cholinv(mean(distribution))
loggammamean(distribution::PointMass) = loggamma(mean(distribution))

Base.getindex(distribution::PointMass, index) = Base.getindex(getpointmass(distribution), index)

Distributions.insupport(distribution::PointMass, x::Real) = x == getpointmass(distribution)
Distributions.pdf(distribution::PointMass, x::Real)       = Distributions.insupport(distribution, x) ? 1.0 : 0.0
Distributions.logpdf(distribution::PointMass, x::Real)    = Distributions.insupport(distribution, x) ? 0.0 : -Inf
Distributions.entropy(::PointMass) = -∞

# Real-based univariate point mass

Distributions.mean(distribution::PointMass{T}) where { T <: Real } = getpointmass(distribution)
Distributions.var(distribution::PointMass{T})  where { T <: Real } = zero(T)
Distributions.std(distribution::PointMass{T})  where { T <: Real } = zero(T)
Distributions.cov(distribution::PointMass{T})  where { T <: Real } = zero(T)

probvec(distribution::PointMass{T})      where { T <: Real } = error("probvec() for univariate point mass is not defined")
logmean(distribution::PointMass{T})      where { T <: Real } = log(mean(distribution))
inversemean(distribution::PointMass{T})  where { T <: Real } = inv(mean(distribution))
loggammamean(distribution::PointMass{T}) where { T <: Real } = loggamma(mean(distribution))

Base.precision(::PointMass{T}) where { T <: Real } = Inf
Base.ndims(::PointMass{T})     where { T <: Real } = 1

# AbstractVector-based multivariate point mass

Distributions.mean(distribution::PointMass{V}) where { T, V <: AbstractVector{T} } = getpointmass(distribution)
Distributions.var(distribution::PointMass{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ))
Distributions.std(distribution::PointMass{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ))
Distributions.cov(distribution::PointMass{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ndims(distribution)))

probvec(distribution::PointMass{V})      where { T, V <: AbstractVector{T} } = mean(distribution)
logmean(distribution::PointMass{V})      where { T, V <: AbstractVector{T} } = log.(mean(distribution))
inversemean(distribution::PointMass{V})  where { T, V <: AbstractVector{T} } = inv.(mean(distribution))
loggammamean(distribution::PointMass{V}) where { T, V <: AbstractVector{T} } = loggamma.(mean(distribution))

Base.precision(distribution::PointMass{V}) where { T, V <: AbstractVector{T} } = one(T) ./ var(distribution)
Base.ndims(distribution::PointMass{V})     where { T, V <: AbstractVector{T} } = length(mean(distribution))

# AbstractMatrix-based matrixvariate point mass

Distributions.mean(distribution::PointMass{M}) where { T, M <: AbstractMatrix{T} } = getpointmass(distribution)
Distributions.var(distribution::PointMass{M})  where { T, M <: AbstractMatrix{T} } = zeros(T, ndims(distribution))
Distributions.std(distribution::PointMass{M})  where { T, M <: AbstractMatrix{T} } = zeros(T, ndims(distribution))
Distributions.cov(distribution::PointMass{M})  where { T, M <: AbstractMatrix{T} } = throw("Distributions.cov(::PointMass{ <: AbstractMatrix }) is not implemented")

probvec(distribution::PointMass{M})      where { T, M <: AbstractMatrix{T} } = error("probvec() for matrix variate point mass is not implemented")
logmean(distribution::PointMass{M})      where { T, M <: AbstractMatrix{T} } = log.(mean(distribution))
inversemean(distribution::PointMass{M})  where { T, M <: AbstractMatrix{T} } = inv.(mean(distribution))
loggammamean(distribution::PointMass{M}) where { T, M <: AbstractMatrix{T} } = loggamma.(mean(distribution))

Base.precision(distribution::PointMass{M}) where { T, M <: AbstractMatrix{T} } = one(T) ./ var(distribution)
Base.ndims(distribution::PointMass{M})     where { T, M <: AbstractMatrix{T} } = size(mean(distribution))
