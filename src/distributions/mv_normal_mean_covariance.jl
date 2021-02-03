export MvNormalMeanCovariance

import Distributions: logdetcov, distrname, sqmahal, sqmahal!, AbstractMvNormal
import LinearAlgebra: diag, Diagonal, dot
import Base: ndims, precision, length, size, prod

struct MvNormalMeanCovariance{ T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T} } <: AbstractMvNormal
    μ :: M
    Σ :: P
end

function MvNormalMeanCovariance(μ::AbstractVector{ <: Real}, Σ::AbstractMatrix{ <: Real }) 
    T = promote_type(eltype(μ), eltype(Σ))
    return MvNormalMeanCovariance(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Σ))
end

MvNormalMeanCovariance(μ::AbstractVector{ <: Integer}, Σ::AbstractMatrix{ <: Integer }) = MvNormalMeanCovariance(float.(μ), float.(Σ))
MvNormalMeanCovariance(μ::AbstractVector, Σ::AbstractVector)                            = MvNormalMeanCovariance(μ, Matrix(Diagonal(Σ)))
MvNormalMeanCovariance(μ::AbstractVector{T}) where T                                    = MvNormalMeanCovariance(μ, convert(AbstractArray{T}, ones(length(μ))))

Distributions.distrname(::MvNormalMeanCovariance) = "MvNormalMeanCovariance"

weightedmean(dist::MvNormalMeanCovariance) = invcov(dist) * mean(dist)

Distributions.mean(dist::MvNormalMeanCovariance)      = dist.μ
Distributions.var(dist::MvNormalMeanCovariance)       = diag(cov(dist))
Distributions.cov(dist::MvNormalMeanCovariance)       = dist.Σ
Distributions.invcov(dist::MvNormalMeanCovariance)    = cholinv(dist.Σ)
Distributions.std(dist::MvNormalMeanCovariance)       = cholsqrt(cov(dist))
Distributions.logdetcov(dist::MvNormalMeanCovariance) = logdet(cov(dist))

Distributions.sqmahal(dist::MvNormalMeanCovariance, x::AbstractVector) = sqmahal!(similar(x), dist, x)

function Distributions.sqmahal!(r, dist::MvNormalMeanCovariance, x::AbstractVector)
    μ = mean(dist)
    for i in 1:length(r)
        @inbounds r[i] = μ[i] - x[i]
    end
    return xT_A_x(r, invcov(dist))
end

Base.eltype(::MvNormalMeanCovariance{T})     where T = T 
Base.precision(dist::MvNormalMeanCovariance)         = invcov(dist)
Base.length(dist::MvNormalMeanCovariance)            = length(mean(dist))
Base.ndims(dist::MvNormalMeanCovariance)             = length(dist)
Base.size(dist::MvNormalMeanCovariance)              = (length(dist), )

function convert(::Type{ <: MvNormalMeanCovariance{T} }, μ::AbstractVector, Σ::AbstractMatrix) where { T <: Real }
    MvNormalMeanCovariance(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Σ))
end

vague(::Type{ <: MvNormalMeanCovariance }, dims::Int) = MvNormalMeanCovariance(zeros(dims), huge .* ones(dims))

function Base.prod(::ProdPreserveParametrisation, left::MvNormalMeanCovariance, right::MvNormalMeanCovariance)
    invcovleft  = invcov(left)
    invcovright = invcov(right)
    Σ = cholinv(invcovleft + invcovright)
    μ = Σ * (invcovleft * mean(left) + invcovright * mean(right))
    return MvNormalMeanCovariance(μ, Σ)
end



