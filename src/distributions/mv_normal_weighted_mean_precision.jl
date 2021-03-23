export MvNormalWeightedMeanPrecision

import Distributions: logdetcov, distrname, sqmahal, sqmahal!, AbstractMvNormal
import LinearAlgebra: diag, Diagonal, dot
import Base: ndims, precision, length, size, prod

struct MvNormalWeightedMeanPrecision{ T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T} } <: AbstractMvNormal
    xi :: M
    Λ  :: P
end

function MvNormalWeightedMeanPrecision(xi::AbstractVector{ <: Real }, Λ::AbstractMatrix{ <: Real }) 
    T = promote_type(eltype(xi), eltype(Λ))
    return MvNormalWeightedMeanPrecision(convert(AbstractArray{T}, xi), convert(AbstractArray{T}, Λ))
end

MvNormalWeightedMeanPrecision(xi::AbstractVector{ <: Integer}, Λ::AbstractMatrix{ <: Integer }) = MvNormalWeightedMeanPrecision(float.(xi), float.(Λ))
MvNormalWeightedMeanPrecision(xi::AbstractVector, Λ::AbstractVector)                            = MvNormalWeightedMeanPrecision(xi, Matrix(Diagonal(Λ)))
MvNormalWeightedMeanPrecision(xi::AbstractVector{T}) where T                                    = MvNormalWeightedMeanPrecision(xi, convert(AbstractArray{T}, ones(length(xi))))

Distributions.distrname(::MvNormalWeightedMeanPrecision) = "MvNormalWeightedMeanPrecision"

weightedmean(dist::MvNormalWeightedMeanPrecision) = dist.xi

Distributions.mean(dist::MvNormalWeightedMeanPrecision)      = cov(dist) * weightedmean(dist)
Distributions.mode(dist::MvNormalWeightedMeanPrecision)      = mean(dist)
Distributions.var(dist::MvNormalWeightedMeanPrecision)       = diag(cov(dist))
Distributions.cov(dist::MvNormalWeightedMeanPrecision)       = cholinv(dist.Λ)
Distributions.invcov(dist::MvNormalWeightedMeanPrecision)    = dist.Λ
Distributions.std(dist::MvNormalWeightedMeanPrecision)       = cholsqrt(cov(dist))
Distributions.logdetcov(dist::MvNormalWeightedMeanPrecision) = -logdet(invcov(dist))

Distributions.sqmahal(dist::MvNormalWeightedMeanPrecision, x::AbstractVector) = sqmahal!(similar(x), dist, x)

function Distributions.sqmahal!(r, dist::MvNormalWeightedMeanPrecision, x::AbstractVector)
    μ = mean(dist)
    for i in 1:length(r)
        @inbounds r[i] = μ[i] - x[i]
    end
    return xT_A_x(r, invcov(dist))
end

Base.eltype(::MvNormalWeightedMeanPrecision{T})     where T = T 
Base.precision(dist::MvNormalWeightedMeanPrecision)         = invcov(dist)
Base.length(dist::MvNormalWeightedMeanPrecision)            = length(weightedmean(dist))
Base.ndims(dist::MvNormalWeightedMeanPrecision)             = length(dist)
Base.size(dist::MvNormalWeightedMeanPrecision)              = (length(dist), )

Base.convert(::Type{ <: MvNormalWeightedMeanPrecision }, xi::AbstractVector, Λ::AbstractMatrix) = MvNormalWeightedMeanPrecision(xi, Λ)

function Base.convert(::Type{ <: MvNormalWeightedMeanPrecision{T} }, xi::AbstractVector, Λ::AbstractMatrix) where { T <: Real }
    MvNormalWeightedMeanPrecision(convert(AbstractArray{T}, xi), convert(AbstractArray{T}, Λ))
end

vague(::Type{ <: MvNormalWeightedMeanPrecision }, dims::Int) = MvNormalWeightedMeanPrecision(zeros(dims), fill(tiny, dims))

function Base.prod(::ProdBestSuitableParametrisation, left::MvNormalWeightedMeanPrecision, right::MvNormalWeightedMeanPrecision)
    return prod(ProdPreserveParametrisation(), left, right)
end

function Base.prod(::ProdPreserveParametrisation, left::MvNormalWeightedMeanPrecision, right::MvNormalWeightedMeanPrecision)
    xi = weightedmean(left) + weightedmean(right) 
    Λ  = invcov(left) + invcov(right)
    return MvNormalWeightedMeanPrecision(xi, Λ)
end