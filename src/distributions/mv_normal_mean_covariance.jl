export MvNormalMeanCovariance

import Distributions: logdetcov, distrname, sqmahal, sqmahal!, AbstractMvNormal
import LinearAlgebra: diag, Diagonal, dot
import Base: ndims, precision, length, size, prod

struct MvNormalMeanCovariance{T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T}} <: AbstractMvNormal
    μ::M
    Σ::P
end

function MvNormalMeanCovariance(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real})
    T = promote_type(eltype(μ), eltype(Σ))
    return MvNormalMeanCovariance(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Σ))
end

function MvNormalMeanCovariance(μ::AbstractVector{<:Integer}, Σ::AbstractMatrix{<:Integer})
    return MvNormalMeanCovariance(float.(μ), float.(Σ))
end

function MvNormalMeanCovariance(μ::AbstractVector{L}, σ::AbstractVector{R}) where {L, R}
    return MvNormalMeanCovariance(μ, convert(Matrix{promote_type(L, R)}, Diagonal(σ)))
end

function MvNormalMeanCovariance(μ::AbstractVector{T}) where {T}
    return MvNormalMeanCovariance(μ, convert(AbstractArray{T}, ones(length(μ))))
end

Distributions.distrname(::MvNormalMeanCovariance) = "MvNormalMeanCovariance"

function weightedmean(dist::MvNormalMeanCovariance)
    z = fastcholesky(cov(dist))
    return z \ mean(dist)
end

function weightedmean_invcov(dist::MvNormalMeanCovariance)
    W = precision(dist)
    xi = W * mean(dist)
    return (xi, W)
end

weightedmean_precision(dist::MvNormalMeanCovariance) = weightedmean_invcov(dist)

Distributions.mean(dist::MvNormalMeanCovariance)      = dist.μ
Distributions.var(dist::MvNormalMeanCovariance)       = diag(cov(dist))
Distributions.cov(dist::MvNormalMeanCovariance)       = dist.Σ
Distributions.invcov(dist::MvNormalMeanCovariance)    = cholinv(dist.Σ)
Distributions.std(dist::MvNormalMeanCovariance)       = cholsqrt(cov(dist))
Distributions.logdetcov(dist::MvNormalMeanCovariance) = chollogdet(cov(dist))

Distributions.sqmahal(dist::MvNormalMeanCovariance, x::AbstractVector) = sqmahal!(similar(x), dist, x)

function Distributions.sqmahal!(r, dist::MvNormalMeanCovariance, x::AbstractVector)
    μ = mean(dist)
    for i in 1:length(r)
        @inbounds r[i] = μ[i] - x[i]
    end
    return xT_A_y(r, invcov(dist), r) # x' * A * x
end

Base.eltype(::MvNormalMeanCovariance{T}) where {T} = T
Base.precision(dist::MvNormalMeanCovariance)       = invcov(dist)
Base.length(dist::MvNormalMeanCovariance)          = length(mean(dist))
Base.ndims(dist::MvNormalMeanCovariance)           = length(dist)
Base.size(dist::MvNormalMeanCovariance)            = (length(dist),)

Base.convert(::Type{<:MvNormalMeanCovariance}, μ::AbstractVector, Σ::AbstractMatrix) = MvNormalMeanCovariance(μ, Σ)

function Base.convert(::Type{<:MvNormalMeanCovariance{T}}, μ::AbstractVector, Σ::AbstractMatrix) where {T <: Real}
    return MvNormalMeanCovariance(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Σ))
end

vague(::Type{<:MvNormalMeanCovariance}, dims::Int) = MvNormalMeanCovariance(zeros(Float64, dims), fill(convert(Float64, huge), dims))

prod_analytical_rule(::Type{<:MvNormalMeanCovariance}, ::Type{<:MvNormalMeanCovariance}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdPreserveType, left::MvNormalMeanCovariance, right::MvNormalMeanCovariance)
    invcovleft  = invcov(left)
    invcovright = invcov(right)
    Σ           = cholinv(invcovleft + invcovright)
    μ           = Σ * (invcovleft * mean(left) + invcovright * mean(right))
    return MvNormalMeanCovariance(μ, Σ)
end

function Base.prod(::ProdAnalytical, left::MvNormalMeanCovariance, right::MvNormalMeanCovariance)
    xi_left, W_left = weightedmean_precision(left)
    xi_right, W_right = weightedmean_precision(right)
    return MvNormalWeightedMeanPrecision(xi_left + xi_right, W_left + W_right)
end

function Base.prod(::ProdAnalytical, left::MvNormalMeanCovariance{T1}, right::MvNormalMeanCovariance{T2}) where {T1 <: LinearAlgebra.BlasFloat, T2 <: LinearAlgebra.BlasFloat}

    # start with parameters of left
    xi, W = weightedmean_precision(left)

    # update W
    W_right = precision(right)
    W .+= W_right

    # update xi without allocating another vector
    T  = promote_type(T1, T2)
    xi = convert(AbstractVector{T}, xi)
    W  = convert(AbstractMatrix{T}, W)
    xi = BLAS.gemv!('N', one(T), convert(AbstractMatrix{T}, W_right), convert(AbstractVector{T}, mean(right)), one(T), xi)

    return MvNormalWeightedMeanPrecision(xi, W)
end
