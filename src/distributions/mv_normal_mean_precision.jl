export MvNormalMeanPrecision

import Distributions: logdetcov, distrname, sqmahal, sqmahal!, AbstractMvNormal
import LinearAlgebra: diag, Diagonal, dot
import Base: ndims, precision, length, size, prod

struct MvNormalMeanPrecision{T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T}} <: AbstractMvNormal
    μ::M
    Λ::P
end

function MvNormalMeanPrecision(μ::AbstractVector{<:Real}, Λ::AbstractMatrix{<:Real})
    T = promote_type(eltype(μ), eltype(Λ))
    return MvNormalMeanPrecision(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Λ))
end

function MvNormalMeanPrecision(μ::AbstractVector{<:Integer}, Λ::AbstractMatrix{<:Integer})
    return MvNormalMeanPrecision(float.(μ), float.(Λ))
end

function MvNormalMeanPrecision(μ::AbstractVector{L}, λ::AbstractVector{R}) where {L, R}
    return MvNormalMeanPrecision(μ, convert(Matrix{promote_type(L, R)}, Diagonal(λ)))
end

function MvNormalMeanPrecision(μ::AbstractVector{T}) where {T}
    return MvNormalMeanPrecision(μ, convert(AbstractArray{T}, ones(length(μ))))
end

Distributions.distrname(::MvNormalMeanPrecision) = "MvNormalMeanPrecision"

weightedmean(dist::MvNormalMeanPrecision) = precision(dist) * mean(dist)

Distributions.mean(dist::MvNormalMeanPrecision)      = dist.μ
Distributions.mode(dist::MvNormalMeanPrecision)      = mean(dist)
Distributions.var(dist::MvNormalMeanPrecision)       = diag(cov(dist))
Distributions.cov(dist::MvNormalMeanPrecision)       = cholinv(dist.Λ)
Distributions.invcov(dist::MvNormalMeanPrecision)    = dist.Λ
Distributions.std(dist::MvNormalMeanPrecision)       = cholsqrt(cov(dist))
Distributions.logdetcov(dist::MvNormalMeanPrecision) = -chollogdet(invcov(dist))

Distributions.sqmahal(dist::MvNormalMeanPrecision, x::AbstractVector) = sqmahal!(similar(x), dist, x)

function Distributions.sqmahal!(r, dist::MvNormalMeanPrecision, x::AbstractVector)
    μ = mean(dist)
    for i in 1:length(r)
        @inbounds r[i] = μ[i] - x[i]
    end
    return dot(r, invcov(dist), r) # x' * A * x
end

Base.eltype(::MvNormalMeanPrecision{T}) where {T} = T
Base.precision(dist::MvNormalMeanPrecision)       = invcov(dist)
Base.length(dist::MvNormalMeanPrecision)          = length(mean(dist))
Base.ndims(dist::MvNormalMeanPrecision)           = length(dist)
Base.size(dist::MvNormalMeanPrecision)            = (length(dist),)

Base.convert(::Type{<:MvNormalMeanPrecision}, μ::AbstractVector, Λ::AbstractMatrix) = MvNormalMeanPrecision(μ, Λ)

function Base.convert(::Type{<:MvNormalMeanPrecision{T}}, μ::AbstractVector, Λ::AbstractMatrix) where {T <: Real}
    MvNormalMeanPrecision(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Λ))
end

vague(::Type{<:MvNormalMeanPrecision}, dims::Int) = MvNormalMeanPrecision(zeros(Float64, dims), fill(convert(Float64, tiny), dims))

prod_analytical_rule(::Type{<:MvNormalMeanPrecision}, ::Type{<:MvNormalMeanPrecision}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdPreserveType, left::MvNormalMeanPrecision, right::MvNormalMeanPrecision)
    Λ = invcov(left) + invcov(right)
    μ = cholinv(Λ) * (invcov(left) * mean(left) + invcov(right) * mean(right))
    return MvNormalMeanPrecision(μ, Λ)
end

function Base.prod(::ProdPreserveType, left::MvNormalMeanPrecision{T1}, right::MvNormalMeanPrecision{T2}) where {T1 <: LinearAlgebra.BlasFloat, T2 <: LinearAlgebra.BlasFloat}
    Λ = precision(left) + precision(right)

    # fast & efficient implementation of precision(right)*mean(right) + precision(left)*mean(left)
    xi = precision(right) * mean(right)
    T  = promote_type(T1, T2)
    xi = convert(AbstractVector{T}, xi)
    Λ  = convert(AbstractMatrix{T}, Λ)
    xi = BLAS.gemv!('N', one(T), convert(AbstractMatrix{T}, precision(left)), convert(AbstractVector{T}, mean(left)), one(T), xi)

    z = fastcholesky(Λ)
    μ = z \ xi

    return MvNormalMeanPrecision(μ, Λ)
end

function Base.prod(::ProdAnalytical, left::MvNormalMeanPrecision, right::MvNormalMeanPrecision)
    W = precision(left) + precision(right)
    xi = precision(left) * mean(left) + precision(right) * mean(right)
    return MvNormalWeightedMeanPrecision(xi, W)
end

function Base.prod(::ProdAnalytical, left::MvNormalMeanPrecision{T1}, right::MvNormalMeanPrecision{T2}) where {T1 <: LinearAlgebra.BlasFloat, T2 <: LinearAlgebra.BlasFloat}
    W = precision(left) + precision(right)

    # fast & efficient implementation of xi = precision(right)*mean(right) + precision(left)*mean(left)
    xi = precision(right) * mean(right)
    T  = promote_type(T1, T2)
    xi = convert(AbstractVector{T}, xi)
    W  = convert(AbstractMatrix{T}, W)
    xi = BLAS.gemv!('N', one(T), convert(AbstractMatrix{T}, precision(left)), convert(AbstractVector{T}, mean(left)), one(T), xi)

    return MvNormalWeightedMeanPrecision(xi, W)
end
