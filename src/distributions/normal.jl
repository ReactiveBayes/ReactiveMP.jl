export GaussianMeanVariance, GaussianMeanPrecision, GaussianWeighteMeanPrecision
export MvGaussianMeanCovariance, MvGaussianMeanPrecision, MvGaussianWeightedMeanPrecision
export UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily, NormalDistributionsFamily
export UnivariateGaussianDistributionsFamily, MultivariateGaussianDistributionsFamily, GaussianDistributionsFamily
export JointNormal, JointGaussian

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

# Joint over multiple Gaussians

"""
    JointNormal

`JointNormal` is an auxilary structure used for the joint marginal over Normally distributed variables.
`JointNormal` stores a vector with the original dimensionalities (ds), so statistics can later be re-separated.

# Fields
- `dist`: joint Normal distribution (typically just a big `MvNormal` distribution)
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
mean_cov(::JointNormal, dist::MvNormalMeanCovariance, ::Tuple) = mean_cov(dist)

# In case if `JointNormal` internal representation stores the actual distribution with a single univariate element we return its statistics as numbers
mean_cov(::JointNormal, dist::MvNormalMeanCovariance, ::Tuple{Tuple{}}) = first.(mean_cov(dist))

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

# We need this function in the unscented still
"""Split a vector in chunks of lengths specified by ds."""
function splitjoint(vec::AbstractVector, ds::Vector{<:Tuple})
    N = length(ds)
    res = Vector{Any}(undef, N)

    d_start = 1
    for k in 1:N # For each original statistic
        d_end = d_start + prod(ds[k]) - 1 # `prod` here returns the dimensionality

        if ds[k] == () # Univariate
            res[k] = vec[d_start] # Return scalar
        else # Multi- of matrix variate
            res[k] = reshape(vec[d_start:d_end], ds[k]) # Return vector or matrix
        end

        d_start = d_end + 1
    end

    return res
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
    start  = sum(prod.(ds[1:index - 1]); init = 0) + 1
    len    = first(sz) 
    stop   = start + len - 1
    μ, Σ = mean_cov(dist)
    # Return the slice of the original `MvNormalMeanCovariance`
    return MvNormalMeanCovariance(view(μ, start:stop), view(Σ, start:stop, start:stop))
end

# `JointNormal` holds a single big gaussian and the dimensionalities are generic, the element is Univariate
function getmarginal(::JointNormal, dist::MvNormalMeanCovariance, ds::Tuple, sz::Tuple{}, index)
    @assert index <= length(ds) "Cannot marginalize `JointNormal` with single entry at index > number of elements"
    start  = sum(prod.(ds[1:index - 1]); init = 0) + 1
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

# Approximation methods extensions for Normal distributions family

# This function extends the `Linearization` approximation method in case if all inputs are from `NormalDistributionsFamily`
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

# collectStatistics

# This functions are still needed for the `Unscented`
function collectStatistics(msgs::Vararg{Any})
    stats = []
    for msg in msgs
        (msg === nothing) && continue # Skip unreported messages
        push!(stats, mean_cov(msg))
    end

    ms = [stat[1] for stat in stats]
    Vs = [stat[2] for stat in stats]
    return (ms, Vs) # Return tuple with vectors for means and covariances
end

function collectStatistics(msg::NormalDistributionsFamily)
    return mean_cov(msg)
end

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
