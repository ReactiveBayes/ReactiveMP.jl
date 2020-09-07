export Dirac

import Distributions: mean, var, cov, std, pdf, logpdf
import Base: ndims, precision
import PDMats: AbstractPDMat

struct Dirac{T}
    point :: T
end

Distributions.pdf(::Dirac)    = throw("pdf(::Dirac) is not implemented")
Distributions.logpdf(::Dirac) = throw("logpdf(::Dirac) is not implemented")

# Real-based univariate dirac's delta

Distributions.mean(distribution::Dirac{T}) where { T <: Real } = distribution.point
Distributions.var(::Dirac{T}) where { T <: Real } = zero(T)
Distributions.std(::Dirac{T}) where { T <: Real } = zero(T)
Distributions.cov(::Dirac{T}) where { T <: Real } = zero(T)

Base.precision(::Dirac{T}) where { T <: Real } = Inf
Base.ndims(::Dirac{T})     where { T <: Real } = 1

Distributions.mean(distribution::Dirac{V}) where { T, V <: AbstractVector{T} } = distribution.point
Distributions.var(distribution::Dirac{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ))
Distributions.std(distribution::Dirac{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ))
Distributions.cov(distribution::Dirac{V})  where { T, V <: AbstractVector{T} } = zeros(T, (ndims(distribution), ndims(distribution)))

Base.precision(distribution::Dirac{V}) where { T, V <: AbstractVector{T} } = one(T) ./ var(distribution)
Base.ndims(distribution::Dirac{V})     where { T, V <: AbstractVector{T} } = length(mean(distribution))

