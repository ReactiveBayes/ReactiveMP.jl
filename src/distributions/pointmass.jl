export PointMass, getpointmass

import Distributions: mean, var, cov, std, insupport, pdf, logpdf, entropy
import Base: ndims, precision, getindex, convert, isapprox
import SpecialFunctions: loggamma, logbeta

struct PointMass{P}
    point :: P
end

variate_form(::PointMass{T})  where { T <: Real }　                = Univariate
variate_form(::PointMass{V})  where { T, V <: AbstractVector{T} }　= Multivariate
variate_form(::PointMass{M})  where { T, M <: AbstractMatrix{T} }　= Matrixvariate

##

getpointmass(distribution::PointMass) = distribution.point

##

Base.getindex(distribution::PointMass, index...) = Base.getindex(getpointmass(distribution), index...)

Distributions.entropy(::PointMass) = -∞

# Real-based univariate point mass

Distributions.insupport(distribution::PointMass{T}, x::Real) where { T <: Real } = x == getpointmass(distribution)
Distributions.pdf(distribution::PointMass{T}, x::Real)       where { T <: Real } = Distributions.insupport(distribution, x) ? 1.0 : 0.0
Distributions.logpdf(distribution::PointMass{T}, x::Real)    where { T <: Real } = Distributions.insupport(distribution, x) ? 0.0 : -Inf

Distributions.mean(distribution::PointMass{T}) where { T <: Real } = getpointmass(distribution)
Distributions.var(distribution::PointMass{T})  where { T <: Real } = zero(T)
Distributions.std(distribution::PointMass{T})  where { T <: Real } = zero(T)
Distributions.cov(distribution::PointMass{T})  where { T <: Real } = zero(T)

probvec(distribution::PointMass{T})         where { T <: Real } = error("probvec(::PointMass{ <: Real }) is not defined")
logmean(distribution::PointMass{T})         where { T <: Real } = log(mean(distribution))
inversemean(distribution::PointMass{T})     where { T <: Real } = inv(mean(distribution))
mirroredlogmean(distribution::PointMass{T}) where { T <: Real } = log(one(T) - mean(distribution))
loggammamean(distribution::PointMass{T})    where { T <: Real } = loggamma(mean(distribution))

Base.precision(::PointMass{T}) where { T <: Real } = Inf
Base.ndims(::PointMass{T})     where { T <: Real } = 1

convert_eltype(::Type{ PointMass }, ::Type{T}, distribution::PointMass{R}) where { T <: Real, R <: Real } = PointMass(convert(T, getpointmass(distribution)))

# AbstractVector-based multivariate point mass

Distributions.insupport(distribution::PointMass{V}, x::AbstractVector) where { T, V <: AbstractVector{T} } = x == getpointmass(distribution)
Distributions.pdf(distribution::PointMass{V}, x::AbstractVector)       where { T, V <: AbstractVector{T} } = Distributions.insupport(distribution, x) ? 1.0 : 0.0
Distributions.logpdf(distribution::PointMass{V}, x::AbstractVector)    where { T, V <: AbstractVector{T} } = Distributions.insupport(distribution, x) ? 0.0 : -Inf

Distributions.mean(distribution::PointMass{V}) where { T, V <: AbstractVector{T} } = getpointmass(distribution)
Distributions.var(distribution::PointMass{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ))
Distributions.std(distribution::PointMass{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ))
Distributions.cov(distribution::PointMass{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ndims(distribution)))

probvec(distribution::PointMass{V})         where { T, V <: AbstractVector{T} } = mean(distribution)
logmean(distribution::PointMass{V})         where { T, V <: AbstractVector{T} } = log.(mean(distribution))
inversemean(distribution::PointMass{V})     where { T, V <: AbstractVector{T} } = error("inversemean(::PointMass{ <: AbstractVector }) is not defined")
mirroredlogmean(distribution::PointMass{V}) where { T, V <: AbstractVector{T} } = error("mirroredlogmean(::PointMass{ <: AbstractVector }) is not defined")
loggammamean(distribution::PointMass{V})    where { T, V <: AbstractVector{T} } = loggamma.(mean(distribution))

Base.precision(distribution::PointMass{V}) where { T, V <: AbstractVector{T} } = one(T) ./ cov(distribution)
Base.ndims(distribution::PointMass{V})     where { T, V <: AbstractVector{T} } = length(mean(distribution))

convert_eltype(::Type{ PointMass }, ::Type{T}, distribution::PointMass{R}) where { T <: Real, R <: AbstractVector }           = PointMass(convert(AbstractVector{T}, getpointmass(distribution)))
convert_eltype(::Type{ PointMass }, ::Type{T}, distribution::PointMass{R}) where { T <: AbstractVector, R <: AbstractVector } = PointMass(convert(T, getpointmass(distribution)))

# AbstractMatrix-based matrixvariate point mass

Distributions.insupport(distribution::PointMass{M}, x::AbstractMatrix) where { T, M <: AbstractMatrix{T} } = x == getpointmass(distribution)
Distributions.pdf(distribution::PointMass{M}, x::AbstractMatrix)       where { T, M <: AbstractMatrix{T} } = Distributions.insupport(distribution, x) ? 1.0 : 0.0
Distributions.logpdf(distribution::PointMass{M}, x::AbstractMatrix)    where { T, M <: AbstractMatrix{T} } = Distributions.insupport(distribution, x) ? 0.0 : -Inf

Distributions.mean(distribution::PointMass{M}) where { T, M <: AbstractMatrix{T} } = getpointmass(distribution)
Distributions.var(distribution::PointMass{M})  where { T, M <: AbstractMatrix{T} } = zeros(T, ndims(distribution))
Distributions.std(distribution::PointMass{M})  where { T, M <: AbstractMatrix{T} } = zeros(T, ndims(distribution))
Distributions.cov(distribution::PointMass{M})  where { T, M <: AbstractMatrix{T} } = error("Distributions.cov(::PointMass{ <: AbstractMatrix }) is not defined")

probvec(distribution::PointMass{M})         where { T, M <: AbstractMatrix{T} } = error("probvec(::PointMass{ <: AbstractMatrix }) is not defined")
logmean(distribution::PointMass{M})         where { T, M <: AbstractMatrix{T} } = log.(mean(distribution))
inversemean(distribution::PointMass{M})     where { T, M <: AbstractMatrix{T} } = cholinv(mean(distribution))
mirroredlogmean(distribution::PointMass{M}) where { T, M <: AbstractMatrix{T} } = error("mirroredlogmean(::PointMass{ <: AbstractMatrix }) is not defined")
loggammamean(distribution::PointMass{M})    where { T, M <: AbstractMatrix{T} } = loggamma.(mean(distribution))

Base.precision(distribution::PointMass{M}) where { T, M <: AbstractMatrix{T} } = one(T) ./ cov(distribution)
Base.ndims(distribution::PointMass{M})     where { T, M <: AbstractMatrix{T} } = size(mean(distribution))

convert_eltype(::Type{ PointMass }, ::Type{T}, distribution::PointMass{R}) where { T <: Real, R <: AbstractMatrix }           = PointMass(convert(AbstractMatrix{T}, getpointmass(distribution)))
convert_eltype(::Type{ PointMass }, ::Type{T}, distribution::PointMass{R}) where { T <: AbstractMatrix, R <: AbstractMatrix } = PointMass(convert(T, getpointmass(distribution)))

Base.isapprox(left::PointMass, right::PointMass; kwargs...) = Base.isapprox(getpointmass(left), getpointmass(right); kwargs...)
Base.isapprox(left::PointMass, right; kwargs...) = false
Base.isapprox(left, right::PointMass; kwargs...) = false