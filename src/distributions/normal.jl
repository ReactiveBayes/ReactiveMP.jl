export GaussianMeanVariance, GaussianMeanPrecision, GaussianWeighteMeanPrecision
export MvGaussianMeanCovariance, MvGaussianMeanPrecision, MvGaussianWeightedMeanPrecision
export UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily, NormalDistributionsFamily
export UnivariateGaussianDistributionsFamily, MultivariateGaussianDistributionsFamily, GaussianDistributionsFamily
export JointNormal, JointGaussian
export UnivariateNormalNaturalParameters, MultivariateNormalNaturalParameters

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

import Base: prod, convert, ndims
import Random: rand!
import Distributions: logpdf
import StatsFuns: invsqrt2π

using LoopVectorization
using LinearAlgebra

# Joint over multiple Gaussians

"""
    JointNormal

`JointNormal` is an auxilary structure used for the joint marginal over Normally distributed variables.
`JointNormal` stores a vector with the original dimensionalities (ds), so statistics can later be re-separated.

# Fields
- `dist`: joint distribution (typically just a big `MvNormal` distribution, but maybe a tuple of individual means and covariance matrices)
- `ds`: a tuple with the original dimensionalities of individual `Normal` distributions
  - `ds[k] = (n,)` where `n` is an integer indicates `Multivariate` normal of size `n`
  - `ds[k] = ()` indicates `Univariate` normal
"""
struct JointNormal{D, S}
    dist :: D
    ds   :: S
end

dimensionalities(joint::JointNormal) = joint.ds

mean_cov(joint::JointNormal) = mean_cov(joint, joint.dist, joint.ds)

# In case if `JointNormal` internal representation stores the actual distribution we simply returns its statistics
mean_cov(::JointNormal, dist::NormalDistributionsFamily, ::Tuple) = mean_cov(dist)

# In case if `JointNormal` internal representation stores the actual distribution with a single univariate element we return its statistics as numbers
mean_cov(::JointNormal, dist::NormalDistributionsFamily, ::Tuple{Tuple{}}) = first.(mean_cov(dist))

# In case if `JointNormal` internal representation stores tuples of means and covariances we need to concatenate them
function mean_cov(::JointNormal, dist::Tuple{Tuple, Tuple}, ds::Tuple)
    total = sum(prod.(ds); init = 0)
    @assert total !== 0 "Broken `JointNormal` state"

    T = promote_type(eltype.(first(dist))..., eltype.(last(dist))...)
    μ = zeros(T, total)
    Σ = zeros(T, total, total)

    sizes = prod.(ds)

    start = 1
    @inbounds for (index, size) in enumerate(sizes)
        dm, dc = first(dist)[index], last(dist)[index]
        μ[start:(start + size - 1)] .= dm
        Σ[start:(start + size - 1), start:(start + size - 1)] .= dc
        start += size
    end

    return (μ, Σ)
end

# In case if `JointNormal` internal representation stores tuples of means and covariances with a single univariate element we return its statistics
function mean_cov(::JointNormal, dist::Tuple{Tuple, Tuple}, ds::Tuple{Tuple})
    return (first(first(dist)), first(last(dist)))
end

entropy(joint::JointNormal) = entropy(joint, joint.dist)

entropy(joint::JointNormal, dist::NormalDistributionsFamily) = entropy(dist)
entropy(joint::JointNormal, dist::Tuple{Tuple, Tuple})       = entropy(convert(MvNormalMeanCovariance, mean_cov(joint)...))

Base.ndims(joint::JointNormal) = ndims(joint, joint.dist)

Base.ndims(joint::JointNormal, dist::NormalDistributionsFamily) = ndims(dist)
Base.ndims(joint::JointNormal, dist::Tuple{Tuple, Tuple})       = sum(length, first(dist))

convert_eltype(::Type{JointNormal}, ::Type{T}, joint::JointNormal) where {T} = convert_eltype(JointNormal, T, joint, joint.dist)

function convert_eltype(::Type{JointNormal}, ::Type{T}, joint::JointNormal, dist::NormalDistributionsFamily) where {T}
    μ, Σ  = map(e -> convert_eltype(T, e), mean_cov(dist))
    cdist = convert(promote_variate_type(variate_form(μ), NormalMeanVariance), μ, Σ)
    return JointNormal(cdist, joint.ds)
end

function Base.convert(::Type{JointNormal}, distribution::UnivariateNormalDistributionsFamily, sizes::Tuple{Tuple{}})
    return JointNormal(distribution, sizes)
end

function Base.convert(::Type{JointNormal}, distribution::MultivariateNormalDistributionsFamily, sizes::Tuple)
    return JointNormal(distribution, sizes)
end

function Base.convert(::Type{JointNormal}, means::Tuple, covs::Tuple)
    @assert length(means) === length(covs) "Cannot create the `JointNormal` with different number of statistics"
    return JointNormal((means, covs), size.(means))
end

"""Return the marginalized statistics of the Gaussian corresponding to an index `index`"""
getmarginal(joint::JointNormal, index) = getmarginal(joint, joint.dist, joint.ds, joint.ds[index], index)

# `JointNormal` holds a single univariate gaussian and the dimensionalities indicate only a single Univariate element
function getmarginal(::JointNormal, dist::NormalMeanVariance, ds::Tuple{Tuple}, sz::Tuple{}, index)
    @assert index === 1 "Cannot marginalize `JointNormal` with single entry at index != 1"
    @assert size(dist) === sz "Broken `JointNormal` state"
    return dist
end

# `JointNormal` holds a single big gaussian and the dimensionalities indicate only a single Multivariate element
function getmarginal(::JointNormal, dist::MvNormalMeanCovariance, ds::Tuple{Tuple}, sz::Tuple{Int}, index)
    @assert index === 1 "Cannot marginalize `JointNormal` with single entry at index != 1"
    @assert size(dist) === sz "Broken `JointNormal` state"
    return dist
end

# `JointNormal` holds a single big gaussian and the dimensionalities indicate only a single Univariate element
function getmarginal(::JointNormal, dist::MvNormalMeanCovariance, ds::Tuple{Tuple}, sz::Tuple{}, index)
    @assert index === 1 "Cannot marginalize `JointNormal` with single entry at index != 1"
    @assert length(dist) === 1 "Broken `JointNormal` state"
    m, V = mean_cov(dist)
    return NormalMeanVariance(first(m), first(V))
end

# `JointNormal` holds a single big gaussian and the dimensionalities are generic, the element is Multivariate
function getmarginal(::JointNormal, dist::MvNormalMeanCovariance, ds::Tuple, sz::Tuple{Int}, index)
    @assert index <= length(ds) "Cannot marginalize `JointNormal` with single entry at index > number of elements"
    start = sum(prod.(ds[1:(index - 1)]); init = 0) + 1
    len   = first(sz)
    stop  = start + len - 1
    μ, Σ  = mean_cov(dist)
    # Return the slice of the original `MvNormalMeanCovariance`
    return MvNormalMeanCovariance(view(μ, start:stop), view(Σ, start:stop, start:stop))
end

# `JointNormal` holds a single big gaussian and the dimensionalities are generic, the element is Univariate
function getmarginal(::JointNormal, dist::MvNormalMeanCovariance, ds::Tuple, sz::Tuple{}, index)
    @assert index <= length(ds) "Cannot marginalize `JointNormal` with single entry at index > number of elements"
    start = sum(prod.(ds[1:(index - 1)]); init = 0) + 1
    μ, Σ = mean_cov(dist)
    # Return the slice of the original `MvNormalMeanCovariance`
    return NormalMeanVariance(μ[start], Σ[start, start])
end

# `JointNormal` holds gaussians individually, simply returns a Multivariate gaussian at index `index`
function getmarginal(::JointNormal, dist::Tuple{Tuple, Tuple}, ds::Tuple, sz::Tuple{Int}, index)
    return MvNormalMeanCovariance(first(dist)[index], last(dist)[index])
end

# `JointNormal` holds gaussians individually, simply returns a Univariate gaussian at index `index`
function getmarginal(::JointNormal, dist::Tuple{Tuple, Tuple}, ds::Tuple, sz::Tuple{}, index)
    return NormalMeanVariance(first(dist)[index], last(dist)[index])
end

# comparing JointNormals - similar to src/distributions/pointmass.jl
Base.isapprox(left::JointNormal, right::JointNormal; kwargs...) = isapprox(left.dist, right.dist; kwargs...) && left.ds == right.ds

"""An alias for the [`JointNormal`](@ref)."""
const JointGaussian = JointNormal

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

function Base.convert(::Type{MvNormalMeanCovariance{T, M}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real, M <: AbstractArray{T}}
    return convert(MvNormalMeanCovariance{T, AbstractArray{T, 1}, AbstractArray{T, 2}}, dist)
end

function Base.convert(::Type{MvNormalMeanCovariance{T, M, P}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real, M <: AbstractArray{T}, P <: AbstractArray{T}}
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

function Base.convert(::Type{MvNormalMeanPrecision{T, M}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real, M <: AbstractArray{T}}
    return convert(MvNormalMeanPrecision{T, AbstractArray{T, 1}, AbstractArray{T, 2}}, dist)
end

function Base.convert(::Type{MvNormalMeanPrecision{T, M, P}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real, M <: AbstractArray{T}, P <: AbstractArray{T}}
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

function Base.convert(::Type{NormalWeightedMeanPrecision{T}}, dist::UnivariateNormalDistributionsFamily) where {T <: Real}
    weightedmean, precision = weightedmean_precision(dist)
    return NormalWeightedMeanPrecision(convert(T, weightedmean), convert(T, precision))
end

function Base.convert(::Type{MvNormalWeightedMeanPrecision{T}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real}
    return convert(MvNormalWeightedMeanPrecision{T, AbstractArray{T, 1}}, dist)
end

function Base.convert(::Type{MvNormalWeightedMeanPrecision{T, M}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real, M <: AbstractArray{T}}
    return convert(MvNormalWeightedMeanPrecision{T, AbstractArray{T, 1}, AbstractArray{T, 2}}, dist)
end

function Base.convert(::Type{MvNormalWeightedMeanPrecision{T, M, P}}, dist::MultivariateNormalDistributionsFamily) where {T <: Real, M <: AbstractArray{T}, P <: AbstractArray{T}}
    weightedmean, precision = weightedmean_precision(dist)
    return MvNormalWeightedMeanPrecision(convert(M, weightedmean), convert(P, precision))
end

function Base.convert(::Type{NormalWeightedMeanPrecision}, dist::UnivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(NormalWeightedMeanPrecision{T}, dist)
end

function Base.convert(::Type{MvNormalWeightedMeanPrecision}, dist::MultivariateNormalDistributionsFamily{T}) where {T <: Real}
    return convert(MvNormalWeightedMeanPrecision{T}, dist)
end

# Basic prod fallbacks to weighted mean precision and converts first argument back

prod_analytical_rule(::Type{<:UnivariateNormalDistributionsFamily}, ::Type{<:UnivariateNormalDistributionsFamily}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::L, right::R) where {L <: UnivariateNormalDistributionsFamily, R <: UnivariateNormalDistributionsFamily}
    wleft  = convert(NormalWeightedMeanPrecision, left)
    wright = convert(NormalWeightedMeanPrecision, right)
    return prod(ProdAnalytical(), wleft, wright)
end

prod_analytical_rule(::Type{<:MultivariateNormalDistributionsFamily}, ::Type{<:MultivariateNormalDistributionsFamily}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::L, right::R) where {L <: MultivariateNormalDistributionsFamily, R <: MultivariateNormalDistributionsFamily}
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

function Random.rand(rng::AbstractRNG, dist::UnivariateNormalDistributionsFamily{T}, size::Int64) where {T}
    container = Vector{T}(undef, size)
    return rand!(rng, dist, container)
end

function Random.rand!(rng::AbstractRNG, dist::UnivariateNormalDistributionsFamily, container::AbstractArray{T}) where {T <: Real}
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

function Random.rand!(rng::AbstractRNG, dist::MultivariateNormalDistributionsFamily, container::AbstractArray{T}) where {T <: Real}
    preallocated = similar(container)
    randn!(rng, reshape(preallocated, length(preallocated)))
    μ, L = mean_std(dist)
    @views for i in axes(preallocated, 2)
        copyto!(container[:, i], μ)
        mul!(container[:, i], L, preallocated[:, i], 1, 1)
    end
    container
end

## Natural parameters for the Normal distribution

### Univariate case

struct UnivariateNormalNaturalParameters{T <: Real} <: NaturalParameters
    weighted_mean::T
    minus_half_precision::T
end

UnivariateNormalNaturalParameters(weighted_mean::Real, minus_half_precision::Real)       = UnivariateNormalNaturalParameters(promote(weighted_mean, minus_half_precision)...)
UnivariateNormalNaturalParameters(weighted_mean::Integer, minus_half_precision::Integer) = UnivariateNormalNaturalParameters(float(weighted_mean), float(minus_half_precision))

function UnivariateNormalNaturalParameters(v::AbstractVector{T}) where {T <: Real}
    @assert length(v) === 2 "`NormalNaturalParameters` must accept a vector of length `2`."
    return UnivariateNormalNaturalParameters(v[1], v[2])
end

Base.convert(::Type{UnivariateNormalNaturalParameters}, weighted_mean::Real, minus_half_precision::Real) =
    convert(UnivariateNormalNaturalParameters{promote_type(typeof(weighted_mean), typeof(minus_half_precision))}, weighted_mean, minus_half_precision)

Base.convert(::Type{UnivariateNormalNaturalParameters{T}}, weighted_mean::Real, minus_half_precision::Real) where {T} =
    UnivariateNormalNaturalParameters(convert(T, weighted_mean), convert(T, minus_half_precision))

Base.convert(::Type{UnivariateNormalNaturalParameters}, vector::AbstractVector) = convert(UnivariateNormalNaturalParameters{eltype(vector)}, vector)

Base.convert(::Type{UnivariateNormalNaturalParameters{T}}, vector::AbstractVector) where {T} = UnivariateNormalNaturalParameters(convert(AbstractVector{T}, vector))

function Base.:(==)(left::UnivariateNormalNaturalParameters, right::UnivariateNormalNaturalParameters)
    return left.weighted_mean == right.weighted_mean && left.minus_half_precision == right.minus_half_precision
end

as_naturalparams(::Type{T}, args...) where {T <: UnivariateNormalNaturalParameters} = convert(UnivariateNormalNaturalParameters, args...)

### Multivariate case

struct MultivariateNormalNaturalParameters{T <: Real, M <: AbstractArray{T}, P <: AbstractMatrix{T}} <: NaturalParameters
    weighted_mean::M
    minus_half_precision_matrix::P
end

function MultivariateNormalNaturalParameters(weighted_mean::AbstractVector{<:Real}, minus_half_precision_matrix::AbstractMatrix{<:Real})
    T = promote_type(eltype(weighted_mean), eltype(minus_half_precision_matrix))
    return MultivariateNormalNaturalParameters(convert(AbstractArray{T}, weighted_mean), convert(AbstractArray{T}, minus_half_precision_matrix))
end

function MultivariateNormalNaturalParameters(weighted_mean::AbstractVector{<:Integer}, minus_half_precision_matrix::AbstractMatrix{<:Integer})
    return MultivariateNormalNaturalParameters(float.(weighted_mean), float.(minus_half_precision_matrix))
end

function MultivariateNormalNaturalParameters(weighted_mean::AbstractVector{T}, minus_half_precision_matrix::AbstractMatrix{T}) where {T <: Real}
    if (length(weighted_mean) !== size(minus_half_precision_matrix, 1)) || (length(weighted_mean) !== size(minus_half_precision_matrix, 2))
        error("`MvNormalNaturalParameters` can not be created from shapes: mean `$(size(weighted_mean))` and matrix `$(size(minus_half_precision_matrix))`.")
    end
    return MultivariateNormalNaturalParameters{T, typeof(weighted_mean), typeof(minus_half_precision_matrix)}(weighted_mean, minus_half_precision_matrix)
end

function MultivariateNormalNaturalParameters(v::AbstractVector{T}) where {T}
    k = length(v)
    d = convert(Int, (-1 + sqrt(4 * k + 1)) / 2)

    @assert (d^2 + d) === k "Vector dimensionality constraints are not fullfiled"

    return MultivariateNormalNaturalParameters(collect(view(v, 1:d)), collect(reshape(view(v, (d + 1):lastindex(v)), d, d)))
end

Base.convert(::Type{MultivariateNormalNaturalParameters}, weighted_mean::AbstractVector, minus_half_precision_matrix::AbstractMatrix) =
    convert(MultivariateNormalNaturalParameters{promote_type(eltype(weighted_mean), eltype(minus_half_precision_matrix))}, weighted_mean, minus_half_precision_matrix)

Base.convert(::Type{MultivariateNormalNaturalParameters{T}}, weighted_mean::AbstractVector, minus_half_precision_matrix::AbstractMatrix) where {T} =
    MultivariateNormalNaturalParameters(convert(AbstractVector{T}, weighted_mean), convert(AbstractMatrix{T}, minus_half_precision_matrix))

Base.convert(::Type{MultivariateNormalNaturalParameters}, vector::AbstractVector) = convert(MultivariateNormalNaturalParameters{eltype(vector)}, vector)

Base.convert(::Type{MultivariateNormalNaturalParameters{T}}, vector::AbstractVector) where {T} = MultivariateNormalNaturalParameters(convert(AbstractVector{T}, vector))

function Base.:(==)(left::MultivariateNormalNaturalParameters, right::MultivariateNormalNaturalParameters)
    return left.weighted_mean == right.weighted_mean && left.minus_half_precision_matrix == right.minus_half_precision_matrix
end

as_naturalparams(::Type{T}, args...) where {T <: MultivariateNormalNaturalParameters} = convert(MultivariateNormalNaturalParameters, args...)

function Base.vec(p::UnivariateNormalNaturalParameters)
    return [p.weighted_mean, p.minus_half_precision]
end

function Base.vec(p::MultivariateNormalNaturalParameters)
    return vcat(p.weighted_mean, vec(p.minus_half_precision_matrix))
end

# Standard parameters to natural parameters
function naturalparams(dist::UnivariateNormalDistributionsFamily)
    weighted_mean, precision = weightedmean_precision(dist)
    return UnivariateNormalNaturalParameters(weighted_mean, -precision / 2)
end

function naturalparams(dist::MultivariateGaussianDistributionsFamily)
    weighted_mean, precision = weightedmean_precision(dist)
    return MultivariateNormalNaturalParameters(weighted_mean, -precision / 2)
end

function convert(::Type{Distribution}, η::UnivariateNormalNaturalParameters)
    return NormalWeightedMeanPrecision(η.weighted_mean, -2 * η.minus_half_precision)
end

function convert(::Type{Distribution}, η::MultivariateNormalNaturalParameters)
    return MvNormalWeightedMeanPrecision(η.weighted_mean, -2 * η.minus_half_precision_matrix)
end

function Base.:+(left::UnivariateNormalNaturalParameters, right::UnivariateNormalNaturalParameters)
    return UnivariateNormalNaturalParameters(left.weighted_mean + right.weighted_mean, left.minus_half_precision + right.minus_half_precision)
end

function Base.:+(left::MultivariateNormalNaturalParameters, right::MultivariateNormalNaturalParameters)
    return MultivariateNormalNaturalParameters(left.weighted_mean .+ right.weighted_mean, left.minus_half_precision_matrix .+ right.minus_half_precision_matrix)
end

function Base.:-(left::UnivariateNormalNaturalParameters, right::UnivariateNormalNaturalParameters)
    return UnivariateNormalNaturalParameters(left.weighted_mean - right.weighted_mean, left.minus_half_precision - right.minus_half_precision)
end

function Base.:-(left::MultivariateNormalNaturalParameters, right::MultivariateNormalNaturalParameters)
    return MultivariateNormalNaturalParameters(left.weighted_mean .- right.weighted_mean, left.minus_half_precision_matrix .- right.minus_half_precision_matrix)
end

function lognormalizer(η::UnivariateNormalNaturalParameters)
    return -η.weighted_mean^2 / (4 * η.minus_half_precision) - log(-2 * η.minus_half_precision) / 2
end

function lognormalizer(η::MultivariateNormalNaturalParameters)
    return η.weighted_mean' * (η.minus_half_precision_matrix \ η.weighted_mean) / 4 + logdet(-2 * η.minus_half_precision_matrix) / 2
end

# Semih: logpdf wrt natural params. ForwardDiff is not stable with reshape function which
# precludes the usage of logPdf functions previously defined. Below function is
# meant to be used with Zygote.
function Distributions.logpdf(η::UnivariateNormalNaturalParameters, x)
    return log(invsqrt2π) + x * η.weighted_mean + x^2 * η.minus_half_precision - lognormalizer(η)
end

function Distributions.logpdf(η::MultivariateNormalNaturalParameters, x)
    ϕx = vcat(x, vec(x * transpose(x)))
    return log((2 * pi)^(-0.5 * length(η.weighted_mean))) + transpose(ϕx) * vec(η) + lognormalizer(η)
end

isproper(params::UnivariateNormalNaturalParameters) = params.minus_half_precision < 0

isproper(params::MultivariateNormalNaturalParameters) = isposdef(-params.minus_half_precision_matrix)

# Approximation methods extensions for Normal distributions family

# This function extends the `Linearization` approximation method in case if all inputs are from the `NormalDistributionsFamily`
function approximate(method::Linearization, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}

    # Collect statistics for the inputs of the function `f`
    statistics = mean_cov.(distributions)
    means      = first.(statistics)
    covs       = last.(statistics)

    # Compute the local approximation for the function `f`
    (A, b) = approximate(method, f, means)

    # Execute the 'joint' message in the linearized version of `f`
    joint       = convert(JointNormal, means, covs)
    jmean, jcov = mean_cov(joint)

    m = A * jmean + b
    V = A * jcov * A'

    return convert(promote_variate_type(variate_form(m), NormalMeanVariance), m, V)
end

# This function extends the `Unscented` approximation method in case if all inputs are from the `NormalDistributionsFamily`
function approximate(method::Unscented, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}
    statistics = mean_cov.(distributions)
    means      = first.(statistics)
    covs       = last.(statistics)

    μ_tilde, Σ_tilde = approximate(method, f, means, covs)

    return convert(promote_variate_type(variate_form(μ_tilde), NormalMeanVariance), μ_tilde, Σ_tilde)
end

# Thes functions extends the `CVI` approximation method in case if input is from the `NormalDistributionsFamily`

get_df_m(::ForwardDiffGrad, ::Type{<:UnivariateNormalNaturalParameters}, ::Type{<:UnivariateGaussianDistributionsFamily}, logp_nc::Function) =
    (z) -> ForwardDiff.derivative(logp_nc, z)

get_df_m(::ForwardDiffGrad, ::Type{<:MultivariateNormalNaturalParameters}, ::Type{<:MultivariateGaussianDistributionsFamily}, logp_nc::Function) =
    (z) -> ForwardDiff.gradient(logp_nc, z)

get_df_v(::ForwardDiffGrad, ::Type{<:UnivariateNormalNaturalParameters}, ::Type{<:UnivariateGaussianDistributionsFamily}, logp_nc::Function, df_m::Function) =
    (z) -> ForwardDiff.derivative(df_m, z)

get_df_v(::ForwardDiffGrad, ::Type{<:MultivariateNormalNaturalParameters}, ::Type{<:MultivariateGaussianDistributionsFamily}, logp_nc::Function, df_m::Function) =
    (z) -> ForwardDiff.jacobian(df_m, z)

function render_cvi(approximation::CVIApproximation, logp_nc::F, initial::GaussianDistributionsFamily) where {F}
    η = naturalparams(initial)
    λ = naturalparams(initial)
    T = typeof(η)

    rng = something(approximation.rng, Random.GLOBAL_RNG)
    opt = approximation.opt
    its = approximation.num_iterations

    df_m = (z) -> get_df_m(approximation.grad, typeof(λ), typeof(initial), logp_nc)(z)
    df_v = (z) -> get_df_v(approximation.grad, typeof(λ), typeof(initial), logp_nc, df_m)(z) / 2

    hasupdated = false

    for _ in 1:its
        q = convert(Distribution, λ)
        z_s = rand(rng, q)
        df_μ1 = df_m(z_s) - 2 * df_v(z_s) * mean(q)
        df_μ2 = df_v(z_s)
        ∇f = as_naturalparams(T, df_μ1, df_μ2)
        ∇ = λ - η - ∇f
        λ_new = as_naturalparams(T, cvi_update!(opt, λ, ∇))
        if isproper(λ_new) && enforce_proper_message(approximation.proper_message, λ_new, η)
            λ = λ_new
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVI approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    return λ
end
