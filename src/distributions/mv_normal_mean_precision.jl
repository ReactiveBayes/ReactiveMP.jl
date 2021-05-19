export MvNormalMeanPrecision

import Distributions: logdetcov, distrname, sqmahal, sqmahal!, AbstractMvNormal
import LinearAlgebra: diag, Diagonal, dot
import Base: ndims, precision, length, size, prod

struct MvNormalMeanPrecision{ T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T} } <: AbstractMvNormal
    μ :: M
    Λ :: P
end

function MvNormalMeanPrecision(μ::AbstractVector{ <: Real }, Λ::AbstractMatrix{ <: Real }) 
    T = promote_type(eltype(μ), eltype(Λ))
    return MvNormalMeanPrecision(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Λ))
end

MvNormalMeanPrecision(μ::AbstractVector{ <: Integer}, Λ::AbstractMatrix{ <: Integer }) = MvNormalMeanPrecision(float.(μ), float.(Λ))
MvNormalMeanPrecision(μ::AbstractVector, Λ::AbstractVector)                            = MvNormalMeanPrecision(μ, Matrix(Diagonal(Λ)))
MvNormalMeanPrecision(μ::AbstractVector{T}) where T                                    = MvNormalMeanPrecision(μ, convert(AbstractArray{T}, ones(length(μ))))

Distributions.distrname(::MvNormalMeanPrecision) = "MvNormalMeanPrecision"

weightedmean(dist::MvNormalMeanPrecision) = precision(dist) * mean(dist)

Distributions.mean(dist::MvNormalMeanPrecision)      = dist.μ
Distributions.mode(dist::MvNormalMeanPrecision)      = mean(dist)
Distributions.var(dist::MvNormalMeanPrecision)       = diag(cov(dist))
Distributions.cov(dist::MvNormalMeanPrecision)       = cholinv(dist.Λ)
Distributions.invcov(dist::MvNormalMeanPrecision)    = dist.Λ
Distributions.std(dist::MvNormalMeanPrecision)       = cholsqrt(cov(dist))
Distributions.logdetcov(dist::MvNormalMeanPrecision) = -logdet(invcov(dist))

Distributions.sqmahal(dist::MvNormalMeanPrecision, x::AbstractVector) = sqmahal!(similar(x), dist, x)

function Distributions.sqmahal!(r, dist::MvNormalMeanPrecision, x::AbstractVector)
    μ = mean(dist)
    for i in 1:length(r)
        @inbounds r[i] = μ[i] - x[i]
    end
    return xT_A_x(r, invcov(dist))
end

Base.eltype(::MvNormalMeanPrecision{T})     where T = T 
Base.precision(dist::MvNormalMeanPrecision)         = invcov(dist)
Base.length(dist::MvNormalMeanPrecision)            = length(mean(dist))
Base.ndims(dist::MvNormalMeanPrecision)             = length(dist)
Base.size(dist::MvNormalMeanPrecision)              = (length(dist), )

Base.convert(::Type{ <: MvNormalMeanPrecision }, μ::AbstractVector, Λ::AbstractMatrix) = MvNormalMeanPrecision(μ, Λ)

function Base.convert(::Type{ <: MvNormalMeanPrecision{T} }, μ::AbstractVector, Λ::AbstractMatrix) where { T <: Real }
    MvNormalMeanPrecision(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Λ))
end

vague(::Type{ <: MvNormalMeanPrecision }, dims::Int) = MvNormalMeanPrecision(zeros(dims), fill(tiny, dims))

prod_analytical_rule(::Type{ <: MvNormalMeanPrecision }, ::Type{ <: MvNormalMeanPrecision }) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::MvNormalMeanPrecision, right::MvNormalMeanPrecision)
    Λ = invcov(left) + invcov(right)
    μ = cholinv(Λ) * (invcov(left) * mean(left) + invcov(right) * mean(right))
    return MvNormalMeanPrecision(μ, Λ)
end



