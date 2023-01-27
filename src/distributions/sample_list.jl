export SampleList, SampleListMeta

import Base: show, ndims, length, size, precision, getindex, broadcasted, map
import Distributions: mean, var, cov, std
import StatsBase: Weights

using StaticArrays
using LoopVectorization

abstract type AbstractSampleListSamplingMethod end

struct BootstrapImportanceSampling <: AbstractSampleListSamplingMethod end

mutable struct SampleListCache{M, C}
    mean           :: M
    cov            :: C
    is_mean_cached :: Bool
    is_cov_cached  :: Bool
end

SampleListCache(::Type{T}, dims::Tuple{}) where {T}         = SampleListCache(zero(T), zero(T), false, false)
SampleListCache(::Type{T}, dims::Tuple{Int}) where {T}      = SampleListCache(zeros(T, first(dims)), zeros(T, first(dims), first(dims)), false, false)
SampleListCache(::Type{T}, dims::Tuple{Int, Int}) where {T} = SampleListCache(zeros(T, dims), zeros(T, prod(dims), prod(dims)), false, false)

is_mean_cached(cache::SampleListCache) = cache.is_mean_cached
is_cov_cached(cache::SampleListCache)  = cache.is_cov_cached

get_mean_storage(cache::SampleListCache) = cache.mean
get_cov_storage(cache::SampleListCache)  = cache.cov

cache_mean!(cache::SampleListCache, mean) = begin
    cache.mean = mean
    cache.is_mean_cached = true
    mean
end
cache_cov!(cache::SampleListCache, cov)   = begin
    cache.cov = cov
    cache.is_cov_cached = true
    cov
end

struct SampleListMeta{W, E, LP, LI}
    unnormalisedweights :: W
    entropy             :: E
    logproposal         :: LP
    logintegrand        :: LI
end

get_unnormalised_weights(meta::SampleListMeta) = meta.unnormalisedweights
get_entropy(meta::SampleListMeta)              = meta.entropy
get_logproposal(meta::SampleListMeta)          = meta.logproposal
get_logintegrand(meta::SampleListMeta)         = meta.logintegrand

call_logproposal(logproposal::Function, x) = logproposal(x)
call_logproposal(logproposal::Any, x)      = logpdf(logproposal, x)

call_logintegrand(logintegrand::Function, x) = logintegrand(x)
call_logintegrand(logintegrand::Any, x)      = logpdf(logintegrand, x)

"""
    SampleList

Generic distribution represented as a list of weighted samples.

# Arguments 
- `samples::S`
- `weights::W`: optional, equivalent to `fill(1 / N, N)` by default, where `N` is the length of `samples` container
"""
struct SampleList{D, S, W, C, M}
    samples :: S
    weights :: W
    cache   :: C
    meta    :: M

    function SampleList(::Val{D}, samples::S, weights::W, meta::M = nothing) where {D, S, W, M}
        @assert div(length(samples), prod(D)) === length(weights) "Invalid sample list samples and weights lengths. `samples` has length $(length(samples)), `weights` has length $(length(weights))"
        @assert eltype(samples) <: Number "Invalid eltype of samples container. Should be a subtype of `Number`, but $(eltype(samples)) has been found. Samples should be stored in a linear one dimensional vector even for multivariate and matrixvariate cases."
        @assert eltype(weights) <: Number "Invalid eltype of weights container. Should be a subtype of `Number`, but $(eltype(weights)) has been found."
        cache = SampleListCache(promote_type(eltype(samples), eltype(weights)), D)
        return new{D, S, W, typeof(cache), M}(samples, weights, cache, meta)
    end
end

Base.show(io::IO, sl::SampleList)         = sample_list_show(io, variate_form(sl), sl)
Base.similar(sl::SampleList{D}) where {D} = SampleList(Val(D), similar(sl.samples), similar(sl.weights))

sample_list_show(io::IO, ::Type{Univariate}, sl::SampleList) = print(io, "SampleList(Univariate, ", length(sl), ")")
sample_list_show(io::IO, ::Type{Multivariate}, sl::SampleList) = print(io, "SampleList(Multivariate(", ndims(sl), "), ", length(sl), ")")
sample_list_show(io::IO, ::Type{Matrixvariate}, sl::SampleList) = print(io, "SampleList(Matrixvariate", ndims(sl), ", ", length(sl), ")")

function SampleList(samples::S) where {S <: AbstractVector}
    N = length(samples)
    return SampleList(samples, fill(one(deep_eltype(S)) / N, N))
end

function SampleList(samples::S, weights::W, meta::M = nothing) where {S, W, M}
    nsamples = length(samples)
    @assert nsamples !== 0 "Empty samples list"
    @assert sum(weights) ≈ one(eltype(weights)) "Weights must sum up to one. sum(weights) = $(sum(weights))"
    D = size(first(samples))
    return SampleList(Val(D), sample_list_linearize(samples, nsamples, prod(D)), weights, meta)
end

const DEFAULT_SAMPLE_LIST_N_SAMPLES = 5000

## Utility functions

Base.eltype(::Type{<:SampleList{D, S, W}}) where {D, S, W} = Tuple{sample_list_eltype(SampleList, D, S), eltype(W)}

sampletype(::SampleList{D, S}) where { D, S } = sample_list_eltype(SampleList, D, S)

sample_list_eltype(::Type{SampleList}, ndims::Tuple{}, ::Type{S}) where {S}         = eltype(S)
sample_list_eltype(::Type{SampleList}, ndims::Tuple{Int}, ::Type{S}) where {S}      = SVector{ndims[1], eltype(S)}
sample_list_eltype(::Type{SampleList}, ndims::Tuple{Int, Int}, ::Type{S}) where {S} = SMatrix{ndims[1], ndims[2], eltype(S), ndims[1] * ndims[2]}

deep_eltype(::Type{<:SampleList{D, S}}) where {D, S} = eltype(S)

## Variate forms

variate_form(::SampleList{D}) where {D} = sample_list_variate_form(D)

sample_list_variate_form(::Tuple{})         = Univariate
sample_list_variate_form(::Tuple{Int})      = Multivariate
sample_list_variate_form(::Tuple{Int, Int}) = Matrixvariate

## Getters

get_weights(sl::SampleList) = get_linear_weights(sl)
get_samples(sl::SampleList) = SamplesOnlyIterator(sl)

get_linear_weights(sl::SampleList) = sl.weights
get_linear_samples(sl::SampleList) = sl.samples
get_cache(sl::SampleList)          = sl.cache
get_meta(sl::SampleList)           = sample_list_check_meta(sl.meta)
is_meta_present(sl::SampleList)    = sl.meta !== nothing

get_data(sl::SampleList) = (length(sl), get_linear_samples(sl), get_linear_weights(sl))

sample_list_check_meta(meta::Any)     = meta
sample_list_check_meta(meta::Nothing) = error("SampleList object has not associated meta information with it.")

get_unnormalised_weights(sl::SampleList) = get_unnormalised_weights(get_meta(sl))
get_entropy(sl::SampleList)              = get_entropy(get_meta(sl))
get_logproposal(sl::SampleList)          = get_logproposal(get_meta(sl))
get_logintegrand(sl::SampleList)         = get_logintegrand(get_meta(sl))

call_logproposal(sl::SampleList, x)  = call_logproposal(get_logproposal(sl), x)
call_logintegrand(sl::SampleList, x) = call_logintegrand(get_logintegrand(sl), x)

Base.length(sl::SampleList) = div(length(get_linear_samples(sl)), prod(ndims(sl)))
Base.ndims(sl::SampleList)  = sample_list_ndims(variate_form(sl), sl)
Base.size(sl::SampleList)   = (length(sl),)

sample_list_ndims(::Type{Univariate}, sl::SampleList{D}) where {D}    = 1
sample_list_ndims(::Type{Multivariate}, sl::SampleList{D}) where {D}  = first(D)
sample_list_ndims(::Type{Matrixvariate}, sl::SampleList{D}) where {D} = D

## Statistics 

# Returns a zeroed container for mean
function sample_list_zero_element(sl::SampleList)
    T = promote_type(eltype(get_linear_weights(sl)), eltype(get_linear_samples(sl)))
    return sample_list_zero_element(variate_form(sl), T, sl)
end

sample_list_zero_element(::Type{Univariate}, ::Type{T}, sl::SampleList) where {T}    = zero(T)
sample_list_zero_element(::Type{Multivariate}, ::Type{T}, sl::SampleList) where {T}  = zeros(T, ndims(sl))
sample_list_zero_element(::Type{Matrixvariate}, ::Type{T}, sl::SampleList) where {T} = zeros(T, ndims(sl))

# Generic mean_cov

mean_cov(sl::SampleList) = sample_list_mean_cov(sl, Val(true))
mean_var(sl::SampleList) = sample_list_mean_var(variate_form(sl), sl)

##

Distributions.mean(sl::SampleList)      = sample_list_mean(sl, Val(true))
Distributions.var(sl::SampleList)       = last(mean_var(sl))
Distributions.cov(sl::SampleList)       = last(mean_cov(sl))
Distributions.invcov(sl::SampleList)    = cholinv(cov(sl))
Distributions.std(sl::SampleList)       = cholsqrt(cov(sl))
Distributions.logdetcov(sl::SampleList) = logdet(cov(sl))

Base.precision(sl::SampleList) = invcov(sl)

function mean_precision(sl::SampleList)
    μ, Σ = mean_cov(sl)
    return μ, cholinv(Σ)
end

function weightedmean_precision(sl::SampleList)
    μ, Λ = mean_precision(sl)
    return Λ * μ, Λ
end

weightedmean(sl::SampleList) = first(weightedmean_precision(sl))

mean(::typeof(log), sl::SampleList)       = sample_list_logmean(variate_form(sl), sl)
mean(::typeof(xtlog), sl::SampleList)     = sample_list_meanlogmean(variate_form(sl), sl)
mean(::typeof(mirrorlog), sl::SampleList) = sample_list_mirroredlogmean(variate_form(sl), sl)

##

# Differential entropy for SampleList
# Entropy is pre-computed during computation of the marginal in `approximate_prod_with_sample_list` function

Distributions.entropy(sl::SampleList) = get_entropy(get_meta(sl))

# `entropy` for the `SampleList` is not defined if `meta` is of type `Nothing`
function Distributions.entropy(::SampleList{D, S, W, C, Nothing}) where {D, S, W, C}
    return CountingReal(deep_eltype(S), -1)
end

## 

vague(::Type{SampleList}; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)                        = sample_list_vague(Univariate, nsamples)
vague(::Type{SampleList}, dims::Int; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)             = sample_list_vague(Multivariate, dims, nsamples)
vague(::Type{SampleList}, dims::Tuple{Int, Int}; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES) = sample_list_vague(Matrixvariate, dims, nsamples)
vague(::Type{SampleList}, dim1::Int, dim2::Int; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)  = sample_list_vague(Matrixvariate, (dim1, dim2), nsamples)

##

sample_list_default_prod_strategy() = BootstrapImportanceSampling()

## prod related stuff
function approximate_prod_with_sample_list(x, y, nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)
    return approximate_prod_with_sample_list(Random.GLOBAL_RNG, x, y, nsamples)
end

function approximate_prod_with_sample_list(rng::AbstractRNG, x, y, nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)
    return approximate_prod_with_sample_list(rng, sample_list_default_prod_strategy(), x, y, nsamples)
end

# `x` is proposal distribution
# `y` is integrand distribution
function approximate_prod_with_sample_list(rng::AbstractRNG, ::BootstrapImportanceSampling, x::Any, y::Any, nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)
    @assert nsamples >= 1 "Number of samples should be non-positive"

    xlogpdf, xsample = logpdf_sample_friendly(x)
    ylogpdf, ysample = logpdf_sample_friendly(y)

    T            = promote_type(eltype(x), eltype(y))
    U            = variate_form(x)
    xsize        = size(x)
    preallocated = preallocate_samples(T, xsize, nsamples)
    samples      = rand!(rng, xsample, reshape(preallocated, (xsize..., nsamples)))

    raw_weights  = Vector{T}(undef, nsamples) # un-normalised
    norm_weights = Vector{T}(undef, nsamples) # normalised

    H_x         = zero(T)
    weights_sum = zero(T)

    for i in 1:nsamples
        # Static indexing from reshaped array
        sample_i = static_getindex(U, xsize, samples, i)
        # Apply log-pdf functions to the samples
        log_sample_x = logpdf(xlogpdf, sample_i)
        log_sample_y = logpdf(ylogpdf, sample_i)

        raw_weight = exp(log_sample_y)

        raw_weights[i]  = raw_weight
        norm_weights[i] = raw_weight # will be renormalised later

        weights_sum += raw_weight
        H_x         += raw_weight * (log_sample_x + log_sample_y)
    end

    # Normalise weights
    @turbo for i in 1:nsamples
        norm_weights[i] /= weights_sum
    end

    # Renormalise H_x
    H_x /= weights_sum

    # Compute the separate contributions to the entropy
    H_y = log(weights_sum) - log(nsamples)
    H_x = -H_x

    entropy = H_x + H_y

    # Inform next step about the proposal and integrand to be used in entropy calculation in smoothing
    logproposal  = xlogpdf
    logintegrand = ylogpdf

    meta = SampleListMeta(raw_weights, entropy, logproposal, logintegrand)

    return SampleList(Val(xsize), preallocated, norm_weights, meta)
end

function approximate_prod_with_sample_list(rng::AbstractRNG, ::AbstractSampleListSamplingMethod, x::SampleList, y::SampleList, nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)
    error("Unsupported SampleList × SampleList prod operation.")
end

function approximate_prod_with_sample_list(rng::AbstractRNG, method::AbstractSampleListSamplingMethod, x::SampleList, y::Any, nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)
    return approximate_prod_with_sample_list(rng, method, y, x, nsamples)
end

# prod of a pdf (or distribution) message and a SampleList message
# this function is capable to calculate entropy with SampleList messages in VMP setting
function approximate_prod_with_sample_list(rng::AbstractRNG, ::BootstrapImportanceSampling, x::Any, y::SampleList{D}, nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES) where {D}

    # TODO: In principle it is possible to implement different prod approximation for different nsamples
    # TODO: This feature would be probably super rare in use so lets postpone it and mark as todo
    @assert length(y) === nsamples "Unsupported SampleList prod approximation: nsamples should match"

    # TODO: Is it possible to support different variate forms here? 
    @assert variate_form(x) === variate_form(y) "Unsupported SampleList prod approximation: variate forms should match"

    # Suppose in the previous time step m1(pdf) and m2(pdf) messages collided.
    # The resulting collision m3 (sampleList) = m1*m2 is supposed to carry
    # the proposal (m1) and integrand (m2) distributions. m1 is the message from which
    # the samples are drawn. m2 is the message on which the samples are evaluated and
    # weights are calculated. In case Particle Filtering (BP), entropy will not be calculated
    # and in the first step there won't be any integrand information.

    xlogpdf, xsample = logpdf_sample_friendly(x)

    log_integrand = if is_meta_present(y) && get_logintegrand(y) !== nothing
        # recall that we are calculating m3*m4. If m3 consists of integrand information
        # update it: new_integrand = m2*m3. This allows us to collide arbitrary number of beliefs
        # to approximate posterior and yet estimate the entropy.
        let y_logintegrand = get_logintegrand(y)
            (sample) -> call_logintegrand(y_logintegrand, sample) + logpdf(xlogpdf, sample)
        end
    else
        # If there is no integrand information before, set it to m4
        xlogpdf
    end

    samples = get_samples(y) # samples come from proposal (m1)
    weights = get_weights(y)

    # Resulting samples and weights will go here
    rcontainer = similar(y)
    # vec here just to convert unusual containers into an array just in case
    # does nothing if `get_weights` returns an array
    rsamples, rweights = get_samples(rcontainer), vec(get_weights(rcontainer))

    rweights_raw      = similar(rweights)
    rweights_prod_sum = zero(eltype(get_weights(y)))

    H_x = zero(eltype(rweights_raw))

    # Compute sample weights
    @inbounds for i in 1:nsamples
        log_sample_x    = logpdf(xlogpdf, samples[i]) # evaluate samples in logm4, i.e. logm4(s)
        raw_weight      = exp(log_sample_x)       # m4(s)
        raw_weight_prod = raw_weight * weights[i] # update the weights of posterior w_unnormalized = m4(s)*w_prev

        rweights_raw[i] = raw_weight
        rweights[i]     = raw_weight_prod

        rweights_prod_sum += raw_weight_prod

        H_x += raw_weight_prod * log_sample_x
    end

    H_x /= rweights_prod_sum

    # Normalize prod weights
    @turbo for i in 1:nsamples
        rweights[i] /= rweights_prod_sum
    end

    # Effective number of particles (Base.Generator is allocation free version of map)
    # neff = sum(rweights .^ 2)
    neff = 1 / mapreduce(abs2, +, rweights)

    # Resample and readjust entropy approximation if required
    if neff < nsamples / 10
        sample!(rng, samples, Weights(weights), rsamples)
        fillv = one(eltype(rweights)) / nsamples
        fill!(rweights, fillv)
        # Readjust H_x after resampling
        H_x = zero(eltype(rweights_raw))
        for i in 1:nsamples
            log_sample_x = logpdf(xlogpdf, rsamples[i])
            H_x += (exp(log_sample_x) * fillv) * log_sample_x
        end
        # H_x /= 1 rweights are normalized at this point, no need to track of its sum
    else
        # Just copy the existing samples instead
        copyto!(get_linear_samples(rsamples), get_linear_samples(samples))
    end

    meta = if is_meta_present(y) && get_logproposal(y) !== nothing && get_unnormalised_weights(y) !== nothing
        y_unnormalised_weights     = get_unnormalised_weights(y)
        r_unnormalised_weights     = similar(y_unnormalised_weights)
        r_unnormalised_weights_sum = zero(eltype(r_unnormalised_weights))
        @inbounds for i in 1:nsamples
            r_unnormalised_weights_prod = y_unnormalised_weights[i] * rweights_raw[i]
            r_unnormalised_weights[i]   = r_unnormalised_weights_prod
            r_unnormalised_weights_sum  += r_unnormalised_weights_prod

            H_x += rweights[i] * (call_logproposal(y, rsamples[i]) + log(y_unnormalised_weights[i]))
        end

        H_y = log(r_unnormalised_weights_sum) - log(nsamples)
        H_x = -H_x

        entropy = H_x + H_y

        SampleListMeta(r_unnormalised_weights, entropy, get_logproposal(y), log_integrand)
    else
        SampleListMeta(nothing, nothing, nothing, log_integrand)
    end

    return SampleList(Val(D), get_linear_samples(rsamples), rweights, meta)
end

############################################################################################
## Everything below this comment is a low-level implementation of SampleList 
## It consists of various routines to compute statistics from internal list implementation
## Here we also implement iteration utilities
## ##########################################################################################

## Lowlevel implementation below...

@inline static_getindex(::Type{Univariate}, ndims::Tuple{}, samples, i)            = samples[i]
@inline static_getindex(::Type{Multivariate}, ndims::Tuple{Int}, samples, i)       = view(samples, :, i)
@inline static_getindex(::Type{Matrixvariate}, ndims::Tuple{Int, Int}, samples, i) = view(samples, :, :, i)

## Preallocation utilities

preallocate_samples(::Type{T}, dims::Tuple, length::Int) where {T} = Vector{T}(undef, length * prod(dims))

## Linearization functions

# Here we cast an array of arrays into a single flat array of floats for better performance
# There is a package for this called ArraysOfArrays.jl, but the performance of handwritten version is way better
# We provide custom optimized mean/cov function for our implementation with LoopVectorization.jl package
function sample_list_linearize end

function sample_list_linearize(samples::AbstractVector{T}, nsamples, size) where {T <: Number}
    return samples
end

function sample_list_linearize(samples::AbstractVector, nsamples, size)
    T = deep_eltype(samples)
    alloc = Vector{T}(undef, nsamples * size)
    for i in 1:nsamples
        copyto!(alloc, (i - 1) * size + 1, samples[i], 1, size)
    end
    return alloc
end

## Cache utilities

sample_list_mean(sl::SampleList, cached)     = sample_list_mean(variate_form(sl), sl, cached)
sample_list_mean_cov(sl::SampleList, cached) = sample_list_mean_cov(variate_form(sl), sl, cached)

# Cache ignoring versions
function sample_list_mean(::Type{U}, sl::SampleList, ::Val{false}) where {U}
    return sample_list_mean!(fill!(similar(get_mean_storage(get_cache(sl))), zero(deep_eltype(sl))), U, sl)
end

function sample_list_mean_cov(::Type{U}, sl::SampleList, ::Val{false}) where {U}
    mean = sample_list_mean(U, sl, Val(false))
    cov = fill!(similar(get_cov_storage(get_cache(sl))), zero(deep_eltype(sl)))
    sample_list_covm!(cov, mean, U, sl)
    return mean, cov
end

# By default we try to save mean in an internal cache
function sample_list_mean(::Type{U}, sl::SampleList, ::Val{true}) where {U}
    cache = get_cache(sl)
    mean  = get_mean_storage(cache)
    # If no cache present, compute and save
    if !is_mean_cached(cache)
        mean = cache_mean!(cache, sample_list_mean!(mean, U, sl))
    end
    return mean
end

# By default we try to save cov in an internal cache
function sample_list_mean_cov(::Type{U}, sl::SampleList, ::Val{true}) where {U}
    cache = get_cache(sl)
    mean  = sample_list_mean(U, sl, Val(true))
    cov   = get_cov_storage(cache)
    # If no cache present, compute and save
    if !is_cov_cached(cache)
        cov = cache_cov!(cache, sample_list_covm!(cov, mean, U, sl))
    end
    return (mean, cov)
end

## Specific implementations

# Compute mean in a preallocated container and return it
function sample_list_mean! end
# Compute covariance with known mean in a preallocated container and return it
function sample_list_covm! end
# Compute mean and variance 
function sample_list_mean_var end
# Compute E[log(x)]
function sample_list_logmean end
# Compute E[xlog(x)]
function sample_list_meanlogmean end
# Compute E[log(1 - x)]
function sample_list_mirroredlogmean end
# Return vague weak-informative sample list
function sample_list_vague end

## Univariate

function sample_list_mean!(μ, ::Type{Univariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    @turbo for i in 1:n
        μ += weights[i] * samples[i]
    end
    return μ
end

function sample_list_covm!(σ², μ, ::Type{Univariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    @turbo for i in 1:n
        σ² += weights[i] * abs2(samples[i] - μ)
    end
    σ² = (n / (n - 1)) * σ²
    return σ²
end

function sample_list_mean_var(::Type{Univariate}, sl::SampleList)
    return sample_list_mean_cov(Univariate, sl, Val(true))
end

function sample_list_logmean(::Type{Univariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    logμ = sample_list_zero_element(sl)
    @turbo for i in 1:n
        logμ += weights[i] * log(samples[i])
    end
    return logμ
end

function sample_list_meanlogmean(::Type{Univariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    μlogμ = sample_list_zero_element(sl)
    @turbo for i in 1:n
        μlogμ += weights[i] * samples[i] * log(samples[i])
    end
    return μlogμ
end

function sample_list_mirroredlogmean(::Type{Univariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    @assert all(0 .<= samples .< 1) "mean of `mirrorlog` of variable does not apply to variables outside of the range [0, 1]"
    mirμ = sample_list_zero_element(sl)
    @turbo for i in 1:n
        mirμ += weights[i] * log(1 - samples[i])
    end
    return mirμ
end

function sample_list_vague(::Type{Univariate}, nsamples::Int)
    targetdist   = vague(Uniform)
    preallocated = preallocate_samples(Float64, (), nsamples)
    rand!(targetdist, preallocated)
    return SampleList(Val(()), preallocated, fill(one(Float64) / nsamples, nsamples), nothing)
end

## Multivariate

function sample_list_mean!(μ, ::Type{Multivariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    k = length(μ)
    @turbo for i in 1:n, j in 1:k
        μ[j] += (weights[i] * samples[(i - 1) * k + j])
    end
    return μ
end

function sample_list_covm!(Σ, μ, ::Type{Multivariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    tmp = similar(μ)
    k = length(tmp)

    @inbounds for i in 1:n
        for j in 1:k
            tmp[j] = samples[(i - 1) * k + j] - μ[j]
        end
        # Fast equivalent of Σ += w .* (tmp * tmp')
        for h in 1:k, l in 1:k
            Σ[(h - 1) * k + l] += weights[i] * tmp[h] * tmp[l]
        end
    end
    s = n / (n - 1)
    @turbo for i in 1:length(Σ)
        Σ[i] *= s
    end
    return Σ
end

function sample_list_mean_var(::Type{Multivariate}, sl::SampleList)
    μ, Σ = sample_list_mean_cov(Multivariate, sl, Val(true))
    return μ, diag(Σ)
end

function sample_list_logmean(::Type{Multivariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    logμ = sample_list_zero_element(sl)
    k = length(logμ)
    @turbo for i in 1:n, j in 1:k
        logμ[j] += (weights[i] * log(samples[(i - 1) * k + j]))
    end
    return logμ
end

function sample_list_meanlogmean(::Type{Multivariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    μlogμ = sample_list_zero_element(sl)
    k = length(μlogμ)
    @turbo for i in 1:n, j in 1:k
        cs = samples[(i - 1) * k + j]
        μlogμ[j] += (weights[i] * cs * log(cs))
    end
    return μlogμ
end

function sample_list_vague(::Type{Multivariate}, dims::Int, nsamples::Int)
    targetdist   = vague(Uniform)
    preallocated = preallocate_samples(Float64, (dims,), nsamples)
    rand!(targetdist, preallocated)
    return SampleList(Val((dims,)), preallocated, fill(one(Float64) / nsamples, nsamples), nothing)
end

## Matrixvariate

function sample_list_mean!(μ, ::Type{Matrixvariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    k = length(μ)
    @turbo for i in 1:n, j in 1:k
        μ[j] += (weights[i] * samples[(i - 1) * k + j])
    end
    return μ
end

function sample_list_covm!(Σ, μ, ::Type{Matrixvariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    k = length(μ)
    rμ = reshape(μ, k)
    tmp = similar(rμ)
    @inbounds for i in 1:n
        for j in 1:k
            tmp[j] = samples[(i - 1) * k + j] - μ[j]
        end
        # Fast equivalent of Σ += w .* (tmp * tmp')
        for h in 1:k, l in 1:k
            Σ[(h - 1) * k + l] += weights[i] * tmp[h] * tmp[l]
        end
    end
    s = n / (n - 1)
    @turbo for i in 1:length(Σ)
        Σ[i] *= s
    end
    return Σ
end

function sample_list_mean_var(::Type{Matrixvariate}, sl::SampleList)
    μ, Σ = sample_list_mean_cov(Matrixvariate, sl, Val(true))
    return μ, reshape(diag(Σ), size(μ))
end

function sample_list_logmean(::Type{Matrixvariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    logμ = sample_list_zero_element(sl)
    k = length(logμ)
    @turbo for i in 1:n, j in 1:k
        logμ[j] += (weights[i] * log(samples[(i - 1) * k + j]))
    end
    return logμ
end

function sample_list_meanlogmean(::Type{Matrixvariate}, sl::SampleList)
    n, samples, weights = get_data(sl)
    μlogμ = sample_list_zero_element(sl)
    k = length(μlogμ)
    @turbo for i in 1:n, j in 1:k
        cs = samples[(i - 1) * k + j]
        μlogμ[j] += (weights[i] * cs * log(cs))
    end
    return μlogμ
end

function sample_list_vague(::Type{Matrixvariate}, dims::Tuple{Int, Int}, nsamples::Int)
    targetdist   = vague(Uniform)
    preallocated = preallocate_samples(Float64, dims, nsamples)
    rand!(targetdist, preallocated)
    return SampleList(Val(dims), preallocated, fill(one(Float64) / nsamples, nsamples), nothing)
end

## Array operations, broadcasting and mapping

struct SamplesOnlyIterator{T, L} <: AbstractVector{T}
    samplelist::L

    function SamplesOnlyIterator(samplelist::L) where {L <: SampleList}
        return new{samples_type(eltype(samplelist)), L}(samplelist)
    end
end

samples_type(::Type{T}) where {L, R, T <: Tuple{L, R}} = L

@inline Base.size(iter::SamplesOnlyIterator) = (length(iter.samplelist),)
@inline Base.getindex(iter::SamplesOnlyIterator, i::Int) = first(getindex(iter.samplelist, i))

@inline function Base.setindex!(iter::SamplesOnlyIterator, v, i::Int)
    samples    = get_linear_samples(iter.samplelist)
    sample_len = prod(ndims(iter.samplelist))
    left       = (i - 1) * sample_len + 1
    right      = left + sample_len - 1
    copyto!(view(samples, left:right), v)
    v
end

get_linear_samples(iter::SamplesOnlyIterator) = get_linear_samples(iter.samplelist)

##

Base.iterate(sl::SampleList)             = (sl[1], 2)
Base.iterate(sl::SampleList, state::Int) = state <= length(sl) ? (sl[state], state + 1) : nothing

@inline Base.getindex(sl::SampleList, i::Int) = sample_list_get_index(variate_form(sl), ndims(sl), sl, i)

@inline function sample_list_get_index(::Type{Univariate}, ndims, sl, i)
    return (get_linear_samples(sl)[i], get_linear_weights(sl)[i])
end

@inline function sample_list_get_index(::Type{Multivariate}, ndims, sl, i)
    samples = get_linear_samples(sl)
    left = (i - 1) * ndims + 1
    right = left + ndims - 1
    # ndims is compile-time here
    return (SVector{ndims}(view(samples, left:right)), get_linear_weights(sl)[i])
end

@inline function sample_list_get_index(::Type{Matrixvariate}, ndims, sl, i)
    p = prod(ndims)
    samples = get_linear_samples(sl)
    left = (i - 1) * p + 1
    right = left + p - 1
    # ndims are compile-time here
    return (SMatrix{ndims[1], ndims[2]}(reshape(view(samples, left:right), ndims)), get_linear_weights(sl)[i])
end

## Transformation routines

transform_samples(f::Function, sl::SampleList) = sample_list_transform_samples(variate_form(sl), f, sl)

@inline input_for_transform(::Type{Univariate}, samples, size, left, right)    = samples[left]
@inline input_for_transform(::Type{Multivariate}, samples, size, left, right)  = SVector{size}(view(samples, left:right))
@inline input_for_transform(::Type{Matrixvariate}, samples, size, left, right) = SMatrix{size[1], size[2]}(reshape(view(samples, left:right), size))

function sample_list_transform_samples(::Type{U}, f::Function, sl::SampleList) where {U}
    n, samples, weights = get_data(sl)
    input_size = ndims(sl)
    input_len = prod(input_size)

    # Here we simulate an original implementation of map function from Julia Base
    # Trick here is to compute the first value so compiler may infer the actual output type
    # Later on output_size and output_len are compile-time constants (given that `f` is type-stable)
    first_item = f(input_for_transform(U, samples, input_size, 1, input_len))

    # After computing first value compiler knows the output type and size of this type
    output_size = size(first_item)
    output_len  = prod(output_size)

    preallocated = preallocate_samples(promote_type(eltype(first_item), eltype(samples)), output_size, n)
    copyto!(view(preallocated, 1:output_len), first_item)

    # We then compute all values from 2 to n into a preallocated buffer
    @views for i in 2:n
        input_left = (i - 1) * input_len + 1
        input_right = input_left + input_len - 1
        output_left = (i - 1) * output_len + 1
        output_right = output_left + output_len - 1
        # We use static matrix size to ensure that we do not allocate extra memory on a heap
        # Instead we try to do all computations on stack as much as possible
        # If `f` function is "bad" and still allocates matrices this optimisation doesn't really work well
        copyto!(preallocated[output_left:output_right], f(input_for_transform(U, samples, input_size, input_left, input_right)))
    end

    # if `f` is type stable and returns StaticArray `output_size` is a compile-time constant
    return SampleList(Val(output_size), preallocated, weights)
end

function transform_weights!(f::Function, sl::SampleList)
    n, _, weights = get_data(sl)
    map!(f, weights, weights)
    norm = sum(weights)
    @turbo for i in 1:n
        weights[i] /= norm
    end
    return sl
end
