export GaussianMeanVariance, GaussianMeanPrecision, GaussianWeighteMeanPrecision
export MvGaussianMeanCovariance, MvGaussianMeanPrecision, MvGaussianWeightedMeanPrecision
export UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily, NormalDistributionsFamily
export UnivariateGaussianDistributionsFamily, MultivariateGaussianDistributionsFamily, GaussianDistributionsFamily

const GaussianMeanVariance            = NormalMeanVariance
const GaussianMeanPrecision           = NormalMeanPrecision
const GaussianWeighteMeanPrecision    = NormalWeightedMeanPrecision
const MvGaussianMeanCovariance        = MvNormalMeanCovariance
const MvGaussianMeanPrecision         = MvNormalMeanPrecision
const MvGaussianWeightedMeanPrecision = MvNormalWeightedMeanPrecision

const UnivariateNormalDistributionsFamily{T}   = Union{NormalMeanPrecision{T}, NormalMeanVariance{T}, NormalWeightedMeanPrecision{T}}
const MultivariateNormalDistributionsFamily{T} = Union{MvNormalMeanPrecision{T}, MvNormalMeanCovariance{T}, MvNormalWeightedMeanPrecision{T}}
const NormalDistributionsFamily{T}             = Union{UnivariateNormalDistributionsFamily{T}, MultivariateNormalDistributionsFamily{T}}

const UnivariateGaussianDistributionsFamily   = UnivariateNormalDistributionsFamily
const MultivariateGaussianDistributionsFamily = MultivariateNormalDistributionsFamily
const GaussianDistributionsFamily             = NormalDistributionsFamily

import Base: prod, convert
import Random: rand!

using LoopVectorization
using StatsFuns: log2π

# Variate forms promotion

promote_variate_type(::Type{Univariate}, ::Type{F}) where {F <: UnivariateNormalDistributionsFamily}     = F
promote_variate_type(::Type{Multivariate}, ::Type{F}) where {F <: MultivariateNormalDistributionsFamily} = F

promote_variate_type(::Type{Univariate}, ::Type{<:MvNormalMeanCovariance})        = NormalMeanVariance
promote_variate_type(::Type{Univariate}, ::Type{<:MvNormalMeanPrecision})         = NormalMeanPrecision
promote_variate_type(::Type{Univariate}, ::Type{<:MvNormalWeightedMeanPrecision}) = NormalWeightedMeanPrecision

promote_variate_type(::Type{Multivariate}, ::Type{<:NormalMeanVariance})          = MvNormalMeanCovariance
promote_variate_type(::Type{Multivariate}, ::Type{<:NormalMeanPrecision})         = MvNormalMeanPrecision
promote_variate_type(::Type{Multivariate}, ::Type{<:NormalWeightedMeanPrecision}) = MvNormalWeightedMeanPrecision

# Conversion to mean - variance parametrisation

function Base.convert(::Type{NormalMeanVariance{T}}, dist::UnivariateNormalDistributionsFamily) where {T <: Real}
    mean, var = mean_var(dist)
    return NormalMeanVariance(convert(T, mean), convert(T, var))
end

function Base.convert(::Type{MvNormalMeanCovariance{T}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real}
    return convert(MvNormalMeanCovariance{T, AbstractArray{T, 1}}, dist)
end

function Base.convert(
    ::Type{MvNormalMeanCovariance{T, M}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}}
    return convert(MvNormalMeanCovariance{T, AbstractArray{T, 1}, AbstractArray{T, 2}}, dist)
end

function Base.convert(
    ::Type{MvNormalMeanCovariance{T, M, P}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}, P <: AbstractArray{T}}
    mean, cov = mean_cov(dist)
    return MvNormalMeanCovariance(convert(M, mean), convert(P, cov))
end

function Base.convert(::Type{NormalMeanVariance}, dist::UnivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(NormalMeanVariance{T}, dist)
end

function Base.convert(::Type{MvNormalMeanCovariance}, dist::MultivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(MvNormalMeanCovariance{T}, dist)
end

# Conversion to mean - precision parametrisation

function Base.convert(::Type{NormalMeanPrecision{T}}, dist::UnivariateNormalDistributionsFamily) where {T <: Real}
    mean, precision = mean_precision(dist)
    return NormalMeanPrecision(convert(T, mean), convert(T, precision))
end

function Base.convert(::Type{MvNormalMeanPrecision{T}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real}
    return convert(MvNormalMeanPrecision{T, AbstractArray{T, 1}}, dist)
end

function Base.convert(
    ::Type{MvNormalMeanPrecision{T, M}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}}
    return convert(MvNormalMeanPrecision{T, AbstractArray{T, 1}, AbstractArray{T, 2}}, dist)
end

function Base.convert(
    ::Type{MvNormalMeanPrecision{T, M, P}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}, P <: AbstractArray{T}}
    mean, precision = mean_precision(dist)
    return MvNormalMeanPrecision(convert(M, mean), convert(P, precision))
end

function Base.convert(::Type{NormalMeanPrecision}, dist::UnivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(NormalMeanPrecision{T}, dist)
end

function Base.convert(::Type{MvNormalMeanPrecision}, dist::MultivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(MvNormalMeanPrecision{T}, dist)
end

# Conversion to weighted mean - precision parametrisation

function Base.convert(
    ::Type{NormalWeightedMeanPrecision{T}},
    dist::UnivariateNormalDistributionsFamily
) where {T <: Real}
    weightedmean, precision = weightedmean_precision(dist)
    return NormalWeightedMeanPrecision(convert(T, weightedmean), convert(T, precision))
end

function Base.convert(
    ::Type{MvNormalWeightedMeanPrecision{T}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real}
    return convert(MvNormalWeightedMeanPrecision{T, AbstractArray{T, 1}}, dist)
end

function Base.convert(
    ::Type{MvNormalWeightedMeanPrecision{T, M}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}}
    return convert(MvNormalWeightedMeanPrecision{T, AbstractArray{T, 1}, AbstractArray{T, 2}}, dist)
end

function Base.convert(
    ::Type{MvNormalWeightedMeanPrecision{T, M, P}},
    dist::MultivariateNormalDistributionsFamily
) where {T <: Real, M <: AbstractArray{T}, P <: AbstractArray{T}}
    weightedmean, precision = weightedmean_precision(dist)
    return MvNormalWeightedMeanPrecision(convert(M, weightedmean), convert(P, precision))
end

function Base.convert(
    ::Type{NormalWeightedMeanPrecision},
    dist::UnivariateNormalDistributionsFamily{T}
) where {T <: Real}
    return convert(NormalWeightedMeanPrecision{T}, dist)
end

function Base.convert(
    ::Type{MvNormalWeightedMeanPrecision},
    dist::MultivariateNormalDistributionsFamily{T}
) where {T <: Real}
    return convert(MvNormalWeightedMeanPrecision{T}, dist)
end

# Basic prod fallbacks to weighted mean precision and converts first argument back

prod_analytical_rule(::Type{<:UnivariateNormalDistributionsFamily}, ::Type{<:UnivariateNormalDistributionsFamily}) =
    ProdAnalyticalRuleAvailable()

function Base.prod(
    ::ProdAnalytical,
    left::L,
    right::R
) where {L <: UnivariateNormalDistributionsFamily, R <: UnivariateNormalDistributionsFamily}
    wleft  = convert(NormalWeightedMeanPrecision, left)
    wright = convert(NormalWeightedMeanPrecision, right)
    return prod(ProdAnalytical(), wleft, wright)
end

function Base.prod(
    ::AddonProdLogScale,
    ::N,
    left::L,
    right::R
) where {N <: UnivariateNormalDistributionsFamily, L <: UnivariateNormalDistributionsFamily, R <: UnivariateNormalDistributionsFamily}
    m_left, v_left   = mean_cov(left)
    m_right, v_right = mean_cov(right)
    v = v_left + v_right
    m = m_left - m_right
    return -(logdet(v) + log2π)/2 - m^2/v/2
end

prod_analytical_rule(::Type{<:MultivariateNormalDistributionsFamily}, ::Type{<:MultivariateNormalDistributionsFamily}) =
    ProdAnalyticalRuleAvailable()

function Base.prod(
    ::ProdAnalytical,
    left::L,
    right::R
) where {L <: MultivariateNormalDistributionsFamily, R <: MultivariateNormalDistributionsFamily}
    wleft  = convert(MvNormalWeightedMeanPrecision, left)
    wright = convert(MvNormalWeightedMeanPrecision, right)
    return prod(ProdAnalytical(), wleft, wright)
end

function Base.prod(
    ::AddonProdLogScale,
    ::N,
    left::L,
    right::R
) where {N <: MultivariateNormalDistributionsFamily, L <: MultivariateNormalDistributionsFamily, R <: MultivariateNormalDistributionsFamily}
    m_left, v_left   = mean_cov(left)
    m_right, v_right = mean_cov(right)
    v = v_left + v_right
    n = length(left)
    v_inv, v_logdet = cholinv_logdet(v)
    m = m_left - m_right
    return -(v_logdet + n*log2π)/2 - dot(m, v_inv, m)/2
end

## Friendly functions

logpdf_sample_friendly(dist::Normal)   = (dist, dist)
logpdf_sample_friendly(dist::MvNormal) = (dist, dist)

function logpdf_sample_friendly(dist::UnivariateNormalDistributionsFamily)
    μ, σ = mean_std(dist)
    friendly = Normal(μ, σ)
    return (friendly, friendly)
end

function logpdf_sample_friendly(dist::MultivariateNormalDistributionsFamily)
    μ, Σ = mean_cov(dist)
    friendly = MvNormal(μ, Σ)
    return (friendly, friendly)
end

# Sample related

## Univariate case

function Random.rand(rng::AbstractRNG, dist::UnivariateNormalDistributionsFamily{T}) where {T}
    μ, σ = mean_std(dist)
    return μ + σ * randn(rng, float(T))
end

function Random.rand(rng::AbstractRNG, dist::UnivariateNormalDistributionsFamily{T}, size::Int64) where {T}
    container = Vector{T}(undef, size)
    return rand!(rng, dist, container)
end

function Random.rand!(
    rng::AbstractRNG,
    dist::UnivariateNormalDistributionsFamily,
    container::AbstractArray{T}
) where {T <: Real}
    randn!(rng, container)
    μ, σ = mean_std(dist)
    @turbo for i in eachindex(container)
        container[i] = μ + σ * container[i]
    end
    container
end

## Multivariate case

function Random.rand(rng::AbstractRNG, dist::MultivariateNormalDistributionsFamily{T}) where {T}
    μ, L = mean_std(dist)
    return μ + L * randn(rng, length(μ))
end

function Random.rand(rng::AbstractRNG, dist::MultivariateNormalDistributionsFamily{T}, size::Int64) where {T}
    container = Matrix{T}(undef, ndims(dist), size)
    return rand!(rng, dist, container)
end

function Random.rand!(
    rng::AbstractRNG,
    dist::MultivariateNormalDistributionsFamily,
    container::AbstractArray{T}
) where {T <: Real}
    preallocated = similar(container)
    randn!(rng, reshape(preallocated, length(preallocated)))
    μ, L = mean_std(dist)
    @views for i in axes(preallocated, 2)
        copyto!(container[:, i], μ)
        mul!(container[:, i], L, preallocated[:, i], 1, 1)
    end
    container
end
