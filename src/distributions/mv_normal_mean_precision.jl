export MvNormalMeanPrecision

import PDMats: AbstractPDMat
import Distributions: mean, var, cov, std, logdetcov, distrname, AbstractMvNormal, logpdf
import LinearAlgebra: diag, Diagonal, UniformScaling
import Base: ndims, precision, length, size, prod

struct MvNormalMeanPrecision{ T <: Real, M <: AbstractVector{T}, P <: AbstractPDMat } <: AbstractMvNormal
    μ :: M
    Λ :: P
end

MvNormalMeanPrecision(μ::AbstractVector{<:Real}, Λ::AbstractMatrix{<:Real}) = MvNormalMeanPrecision(μ, PDMat(Λ))
MvNormalMeanPrecision(μ::AbstractVector{<:Real}, Λ::Diagonal{<:Real})       = MvNormalMeanPrecision(μ, PDiagMat(diag(Λ)))
MvNormalMeanPrecision(μ::AbstractVector{<:Real}, Λ::UniformScaling{<:Real}) = MvNormalMeanPrecision(μ, ScalMat(length(μ), Λ.λ))

Distributions.distrname(::MvNormalMeanPrecision) = "MvNormalMeanPrecision"

Distributions.mean(dist::MvNormalMeanPrecision)      = dist.μ
Distributions.var(dist::MvNormalMeanPrecision)       = diag(cov(dist))
Distributions.cov(dist::MvNormalMeanPrecision)       = inv(dist.Λ)
Distributions.std(dist::MvNormalMeanPrecision)       = sqrt(Matrix(cov(dist)))

# TODO
Distributions.logpdf(dist::MvNormalMeanPrecision, x::AbstractVector{ <: Real }) = logpdf(MvNormal(mean(dist), cov(dist)), x)

Base.precision(dist::MvNormalMeanPrecision) = dist.Λ
Base.length(dist::MvNormalMeanPrecision)    = length(mean(dist))
Base.ndims(dist::MvNormalMeanPrecision)     = length(dist)
Base.size(dist::MvNormalMeanPrecision)      = (length(dist), )

Distributions.logdetcov(dist::MvNormalMeanPrecision) = logdet(cov(dist))

function Base.prod(::ProdPreserveParametrisation, left::MvNormalMeanPrecision, right::MvNormalMeanPrecision)
    Λ = precision(left) + precision(right)
    μ = inv(Λ) * (precision(left) * mean(left) + precision(right) * mean(right))
    return MvNormalMeanPrecision(μ, Λ)
end



