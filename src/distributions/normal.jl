export GaussianMeanVariance, GaussianMeanPrecision, GaussianWeighteMeanPrecision
export MvGaussianMeanCovariance, MvGaussianMeanPrecision, MvGaussianWeightedMeanPrecision
export UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily, NormalDistributionsFamily
export UnivariateGaussianDistributionsFamily, MultivariateGaussianDistributionsFamily, GaussianDistributionsFamily
export NormalNaturalParametrs, MvNormalNaturalParametrs, naturalParams

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

import Base
import Base: prod, convert
import Random: rand!

using LoopVectorization

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

function Random.rand!(
    rng::AbstractRNG,
    dist::UnivariateNormalDistributionsFamily,
    container::AbstractArray{T}
) where {T <: Real}
    randn!(rng, container)
    μ, σ = mean_std(dist)
    @turbo for i in 1:length(container)
        container[i] = μ + σ * container[i]
    end
    container
end

## Multivariate case

function Random.rand(rng::AbstractRNG, dist::MultivariateNormalDistributionsFamily{T}) where {T}
    μ, L = mean_std(dist)
    return μ + L * randn(rng, length(μ))
end

function Random.rand!(
    rng::AbstractRNG,
    dist::MultivariateNormalDistributionsFamily,
    container::AbstractArray{T}
) where {T <: Real}
    preallocated = similar(container)
    randn!(rng, reshape(preallocated, length(preallocated)))
    μ, L = mean_std(dist)
    @views for i in 1:size(preallocated)[2]
        copyto!(container[:, i], μ)
        mul!(container[:, i], L, preallocated[:, i], 1, 1)
    end
    container
end

struct NormalNaturalParametrs <: NaturalParametrs
    weighted_mean
    precision
end

struct MvNormalNaturalParametrs <: NaturalParametrs
    weighted_mean
    precesion_matrix
end

function NormalNaturalParametrs(v)
    return NormalNaturalParametrs(v[1], v[2])
end

function Base.vec(p::NormalNaturalParametrs)
    return [p.weighted_mean, p.precision]
end

# Standard parameters to natural parameters
function naturalParams(dist::UnivariateNormalDistributionsFamily)
    weighted_mean, precision = weightedmean_precision(dist)
    return NormalNaturalParametrs(weighted_mean, -0.5 * precision)
end

function naturalParams(dist::MultivariateGaussianDistributionsFamily)
    weighted_mean, precision = weightedmean_precision(dist)
    MvNormalNaturalParametrs(weighted_mean, -0.5 * precision)
end

function standardDist(η::NormalNaturalParametrs)
    return GaussianWeighteMeanPrecision(η.weighted_mean, -2 * η.precision)
end

function Base.:+(left::NormalNaturalParametrs, right::NormalNaturalParametrs)
    return NormalNaturalParametrs(left.weighted_mean + right.weighted_mean, left.precision + right.precision)
end

function Base.:+(left::MvNormalNaturalParametrs, right::MvNormalNaturalParametrs)
    return NormalNaturalParametrs(left.weighted_mean .+ right.weighted_mean, left.precesion_matrix .+ right.precesion_matrix)
end

function Base.:-(left::NormalNaturalParametrs, right::NormalNaturalParametrs)
    return NormalNaturalParametrs(left.weighted_mean - right.weighted_mean, left.precision - right.precision)
end

function Base.:-(left::MvNormalNaturalParametrs, right::MvNormalNaturalParametrs)
    return NormalNaturalParametrs(left.weighted_mean .- right.weighted_mean, left.precesion_matrix .- right.precesion_matrix)
end

function logNormalizer(η::NormalNaturalParametrs)
    return -(η.weighted_mean^2) / (4 * η.precision) - 0.5 * log(-2 * η.precision)
end

# logPdf wrt natural params. ForwardDiff is not stable with reshape function which
# precludes the usage of logPdf functions previously defined. Below function is
# meant to be used with Zygote.
function logPdf(η::NormalNaturalParametrs, x)
    return log(1 / sqrt(2 * pi)) + x * η.weighted_mean + x^2 * η.precision - logNormalizer(η)
end
