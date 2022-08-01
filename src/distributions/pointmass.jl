export PointMass, getpointmass

import Distributions: mean, var, cov, std, insupport, pdf, logpdf, entropy
import Base: ndims, precision, getindex, convert, isapprox, eltype
import SpecialFunctions: loggamma, logbeta

struct PointMass{P}
    point::P
end

variate_form(::PointMass{T}) where {T <: Real}                 = Univariate
variate_form(::PointMass{V}) where {T, V <: AbstractVector{T}} = Multivariate
variate_form(::PointMass{M}) where {T, M <: AbstractMatrix{T}} = Matrixvariate

##

getpointmass(distribution::PointMass) = distribution.point

##

Base.getindex(distribution::PointMass, index...) = Base.getindex(getpointmass(distribution), index...)

Distributions.entropy(distribution::PointMass) = InfCountingReal(eltype(distribution), -1)

# Real-based univariate point mass

Distributions.insupport(distribution::PointMass{T}, x::Real) where {T <: Real} = x == getpointmass(distribution)
Distributions.pdf(distribution::PointMass{T}, x::Real) where {T <: Real}       = Distributions.insupport(distribution, x) ? one(T) : zero(T)
Distributions.logpdf(distribution::PointMass{T}, x::Real) where {T <: Real}    = Distributions.insupport(distribution, x) ? zero(T) : convert(T, -Inf)

Distributions.mean(distribution::PointMass{T}) where {T <: Real} = getpointmass(distribution)
Distributions.mode(distribution::PointMass{T}) where {T <: Real} = mean(distribution)
Distributions.var(distribution::PointMass{T}) where {T <: Real}  = zero(T)
Distributions.std(distribution::PointMass{T}) where {T <: Real}  = zero(T)
Distributions.cov(distribution::PointMass{T}) where {T <: Real}  = zero(T)

probvec(distribution::PointMass{T}) where {T <: Real} = error("probvec(::PointMass{ <: Real }) is not defined")

mean(fn::F, distribution::PointMass{T}) where {T <: Real, F <: Function} = fn(mean(distribution))

Base.precision(::PointMass{T}) where {T <: Real} = convert(T, Inf)
Base.ndims(::PointMass{T}) where {T <: Real}     = 1

convert_eltype(::Type{PointMass}, ::Type{T}, distribution::PointMass{R}) where {T <: Real, R <: Real} =
    PointMass(convert(T, getpointmass(distribution)))

Base.eltype(::PointMass{T}) where {T <: Real} = T

# AbstractVector-based multivariate point mass

Distributions.insupport(distribution::PointMass{V}, x::AbstractVector) where {T <: Real, V <: AbstractVector{T}} = x == getpointmass(distribution)
Distributions.pdf(distribution::PointMass{V}, x::AbstractVector) where {T <: Real, V <: AbstractVector{T}}       = Distributions.insupport(distribution, x) ? one(T) : zero(T)
Distributions.logpdf(distribution::PointMass{V}, x::AbstractVector) where {T <: Real, V <: AbstractVector{T}}    = Distributions.insupport(distribution, x) ? zero(T) : convert(T, -Inf)

Distributions.mean(distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}} = getpointmass(distribution)
Distributions.mode(distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}} = mean(distribution)
Distributions.var(distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}}  = zeros(T, (ndims(distribution),))
Distributions.std(distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}}  = zeros(T, (ndims(distribution),))
Distributions.cov(distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}}  = zeros(T, (ndims(distribution), ndims(distribution)))

probvec(distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}} = mean(distribution)

mean(::typeof(inv), distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}}       = error("mean of inverse of `::PointMass{ <: AbstractVector }` is not defined")
mean(::typeof(log), distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}}       = log.(mean(distribution))
mean(::typeof(clamplog), distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}}  = clamplog.(mean(distribution))
mean(::typeof(mirrorlog), distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}} = error("mean of mirrorlog of `::PointMass{ <: AbstractVector }` is not defined")
mean(::typeof(loggamma), distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}}  = loggamma.(mean(distribution))
mean(::typeof(logdet), distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}}    = error("mean of logdet of `::PointMass{ <: AbstractVector }` is not defined")

Base.precision(distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}} = one(T) ./ cov(distribution)
Base.ndims(distribution::PointMass{V}) where {T <: Real, V <: AbstractVector{T}}     = length(mean(distribution))

convert_eltype(::Type{PointMass}, ::Type{T}, distribution::PointMass{R}) where {T <: Real, R <: AbstractVector}           = PointMass(convert(AbstractVector{T}, getpointmass(distribution)))
convert_eltype(::Type{PointMass}, ::Type{T}, distribution::PointMass{R}) where {T <: AbstractVector, R <: AbstractVector} = PointMass(convert(T, getpointmass(distribution)))

Base.eltype(::PointMass{V}) where {T <: Real, V <: AbstractVector{T}} = T

# AbstractMatrix-based matrixvariate point mass

Distributions.insupport(distribution::PointMass{M}, x::AbstractMatrix) where {T <: Real, M <: AbstractMatrix{T}} = x == getpointmass(distribution)
Distributions.pdf(distribution::PointMass{M}, x::AbstractMatrix) where {T <: Real, M <: AbstractMatrix{T}}       = Distributions.insupport(distribution, x) ? one(T) : zero(T)
Distributions.logpdf(distribution::PointMass{M}, x::AbstractMatrix) where {T <: Real, M <: AbstractMatrix{T}}    = Distributions.insupport(distribution, x) ? zero(T) : convert(T, -Inf)

Distributions.mean(distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}} = getpointmass(distribution)
Distributions.mode(distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}} = mean(distribution)
Distributions.var(distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}}  = zeros(T, ndims(distribution))
Distributions.std(distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}}  = zeros(T, ndims(distribution))
Distributions.cov(distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}}  = error("Distributions.cov(::PointMass{ <: AbstractMatrix }) is not defined")

probvec(distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}} =
    error("probvec(::PointMass{ <: AbstractMatrix }) is not defined")

mean(::typeof(inv), distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}}       = cholinv(mean(distribution))
mean(::typeof(log), distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}}       = log.(mean(distribution))
mean(::typeof(clamplog), distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}}  = clamplog.(mean(distribution))
mean(::typeof(mirrorlog), distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}} = error("mean of mirrorlog of `::PointMass{ <: AbstractMatrix }` is not defined")
mean(::typeof(loggamma), distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}}  = loggamma.(mean(distribution))
mean(::typeof(logdet), distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}}    = logdet(mean(distribution))

Base.precision(distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}} = one(T) ./ cov(distribution)
Base.ndims(distribution::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}}     = size(mean(distribution))

convert_eltype(::Type{PointMass}, ::Type{T}, distribution::PointMass{R}) where {T <: Real, R <: AbstractMatrix}           = PointMass(convert(AbstractMatrix{T}, getpointmass(distribution)))
convert_eltype(::Type{PointMass}, ::Type{T}, distribution::PointMass{R}) where {T <: AbstractMatrix, R <: AbstractMatrix} = PointMass(convert(T, getpointmass(distribution)))

Base.eltype(::PointMass{M}) where {T <: Real, M <: AbstractMatrix{T}} = T

Base.isapprox(left::PointMass, right::PointMass; kwargs...) =
    Base.isapprox(getpointmass(left), getpointmass(right); kwargs...)
Base.isapprox(left::PointMass, right; kwargs...) = false
Base.isapprox(left, right::PointMass; kwargs...) = false