export Dirac, getpointmass

import Distributions: Dirac, mean, var, cov, std, pdf, logpdf, entropy
import Base: ndims, precision, getindex

getpointmass(dist::Dirac) = mean(dist)

logmean(dist::Dirac)     = log(mean(dist))
inversemean(dist::Dirac) = cholinv(mean(dist))

Base.getindex(dist::Dirac, index) = Base.getindex(getpointmass(dist), index)

score(::DifferentialEntropy, ::Marginal{ <: Dirac }) = -∞

# Real-based univariate dirac's delta

Base.precision(::Dirac{T}) where { T <: Real } = Inf
Base.ndims(::Dirac{T})     where { T <: Real } = 1

# AbstractVector-based multivariate dirac's delta

Distributions.var(distribution::Dirac{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ))
Distributions.std(distribution::Dirac{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ))
Distributions.cov(distribution::Dirac{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ndims(distribution)))

Base.precision(distribution::Dirac{V}) where { T, V <: AbstractVector{T} } = one(T) ./ var(distribution)
Base.ndims(distribution::Dirac{V})     where { T, V <: AbstractVector{T} } = length(mean(distribution))

# AbstractMatrix-based matrixvariate dirac's delta

Distributions.var(distribution::Dirac{M})  where { T, M <: AbstractMatrix{T} } = zeros(T, ndims(distribution))
Distributions.std(distribution::Dirac{M})  where { T, M <: AbstractMatrix{T} } = zeros(T, ndims(distribution))
Distributions.cov(distribution::Dirac{M})  where { T, M <: AbstractMatrix{T} } = throw("Distributions.cov(::Dirac{ <: AbstractMatrix }) is not implemented")

Base.precision(distribution::Dirac{M}) where { T, M <: AbstractMatrix{T} } = one(T) ./ var(distribution)
Base.ndims(distribution::Dirac{M})     where { T, M <: AbstractMatrix{T} } = size(mean(distribution))
