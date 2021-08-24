export SampleList, SampleListMeta

import Base: show, ndims, length, size, precision, getindex, broadcasted, map
import Distributions: mean, var, cov, std

using StaticArrays
using LoopVectorization

mutable struct SampleListCache{M, C}
    mean :: M
    cov  :: C
    is_mean_cached :: Bool
    is_cov_cached  :: Bool
end

SampleListCache(::Type{T}, dims::Tuple{})         where T = SampleListCache(zero(T), zero(T), false, false)
SampleListCache(::Type{T}, dims::Tuple{Int})      where T = SampleListCache(zeros(T, first(dims)), zeros(T, first(dims), first(dims)), false, false)
SampleListCache(::Type{T}, dims::Tuple{Int, Int}) where T = SampleListCache(zeros(T, dims), zeros(T, prod(dims), prod(dims)), false, false)

is_mean_cached(cache::SampleListCache) = cache.is_mean_cached
is_cov_cached(cache::SampleListCache)  = cache.is_cov_cached

get_mean_storage(cache::SampleListCache) = cache.mean
get_cov_storage(cache::SampleListCache)  = cache.cov

cache_mean!(cache::SampleListCache, mean) = begin cache.mean = mean; cache.is_mean_cached = true; mean end
cache_cov!(cache::SampleListCache, cov)   = begin cache.cov  = cov; cache.is_cov_cached = true; cov end

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

    function SampleList(::Val{D}, samples::S, weights::W, meta::M = nothing) where { D, S, W, M }
        @assert div(length(samples), prod(D)) === length(weights) "Invalid sample list samples and weights lengths. `samples` has length $(length(samples)), `weights` has length $(length(weights))"
        @assert eltype(samples) <: Number "Invalid eltype of samples container. Should be a subtype of `Number`, but $(eltype(samples)) has been found. Samples should be stored in a linear one dimensional vector even for multivariate and matrixvariate cases."
        @assert eltype(weights) <: Number "Invalid eltype of weights container. Should be a subtype of `Number`, but $(eltype(weights)) has been found."
        cache = SampleListCache(promote_type(eltype(samples), eltype(weights)), D)
        return new{D, S, W, typeof(cache), M}(samples, weights, cache, meta)
    end
end

Base.show(io::IO, sl::SampleList) = sample_list_show(io, variate_form(sl), sl)

sample_list_show(io::IO, ::Type{Univariate}, sl::SampleList) = print(io, "SampleList(Univariate, ", length(sl), ")")
sample_list_show(io::IO, ::Type{Multivariate}, sl::SampleList) = print(io, "SampleList(Multivariate(", ndims(sl),"), ", length(sl), ")")
sample_list_show(io::IO, ::Type{Matrixvariate}, sl::SampleList) = print(io, "SampleList(Matrixvariate", ndims(sl), ", ", length(sl), ")")

function SampleList(samples::S) where { S <: AbstractVector } 
    return SampleList(samples, OneDivNVector(deep_eltype(S), length(samples)))
end

function SampleList(samples::S, weights::W, meta::M = nothing) where { S, W, M }
    nsamples = length(samples)
    @assert nsamples !== 0 "Empty samples list"
    D = size(first(samples))
    return SampleList(Val(D), sample_list_linearize(samples, nsamples, prod(D)), weights, meta)
end

const DEFAULT_SAMPLE_LIST_N_SAMPLES = 5000

## Utility functions

Base.eltype(::Type{ <: SampleList{D, S, W} }) where { D, S, W } = Tuple{ sample_list_eltype(SampleList, D, S), eltype(W) }

sample_list_eltype(::Type{ SampleList }, ndims::Tuple{}, ::Type{S}) where S         = eltype(S)
sample_list_eltype(::Type{ SampleList }, ndims::Tuple{Int}, ::Type{S}) where S      = SVector{ ndims[1], eltype(S) }
sample_list_eltype(::Type{ SampleList }, ndims::Tuple{Int, Int}, ::Type{S}) where S = SMatrix{ ndims[1], ndims[2], eltype(S), ndims[1] * ndims[2] }

deep_eltype(::Type{ <: SampleList{ D, S } }) where { D, S } = eltype(S)

## Linearization functions

# Here we cast an array of arrays into a single flat array of floats for better performance
# There is a package for this called ArraysOfArrays.jl, but the performance of handwritten version is way better
# We provide custom optimized mean/cov function for our implementation with LoopVectorization.jl package
function sample_list_linearize end

function sample_list_linearize(samples::AbstractVector{T}, nsamples, size) where { T <: Number } 
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

## Variate forms

variate_form(::SampleList{ D }) where { D } = sample_list_variate_form(D)

sample_list_variate_form(::Tuple{})         = Univariate
sample_list_variate_form(::Tuple{Int})      = Multivariate
sample_list_variate_form(::Tuple{Int, Int}) = Matrixvariate

## Getters

get_weights(sl::SampleList) = get_linear_weights(sl)
get_samples(sl::SampleList) = error("SampleList `get_samples` is forbidden. Use broadcasting or map operations to transform sample list")

get_linear_weights(sl::SampleList) = sl.weights
get_linear_samples(sl::SampleList) = sl.samples
get_cache(sl::SampleList)   = sl.cache
get_meta(sl::SampleList)    = sample_list_check_meta(sl.meta)

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
Base.size(sl::SampleList)   = (length(sl), )

sample_list_ndims(::Type{ Univariate }, sl::SampleList{ D }) where { D }    = 1
sample_list_ndims(::Type{ Multivariate }, sl::SampleList{ D }) where { D }  = first(D)
sample_list_ndims(::Type{ Matrixvariate }, sl::SampleList{ D }) where { D } = D

## Statistics 

# Returns a zeroed container for mean
function sample_list_zero_element(sl::SampleList) 
    T = promote_type(eltype(get_linear_weights(sl)), eltype(get_linear_samples(sl)))
    return sample_list_zero_element(variate_form(sl), T, sl)
end

sample_list_zero_element(::Type{ Univariate }, ::Type{T}, sl::SampleList) where T   = zero(T)
sample_list_zero_element(::Type{ Multivariate }, ::Type{T}, sl::SampleList) where T = zeros(T, ndims(sl))
sample_list_zero_element(::Type{ Matrixvariate }, ::Type{T}, sl::SampleList) where T = zeros(T, ndims(sl))

# Generic mean_cov

mean_cov(sl::SampleList) = sample_list_mean_cov(variate_form(sl), sl)
mean_var(sl::SampleList) = sample_list_mean_var(variate_form(sl), sl)

##

Distributions.mean(sl::SampleList)      = sample_list_mean(variate_form(sl), sl)
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

weightedmean(sl::SampleList)    = first(weightedmean_precision(sl))
logmean(sl::SampleList)         = sample_list_logmean(variate_form(sl), sl)
meanlogmean(sl::SampleList)     = sample_list_meanlogmean(variate_form(sl), sl)
mirroredlogmean(sl::SampleList) = sample_list_mirroredlogmean(variate_form(sl), sl)

## 

vague(::Type{ SampleList }; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)                        = sample_list_vague(Univariate, nsamples)
vague(::Type{ SampleList }, dims::Int; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)             = sample_list_vague(Multivariate, dims, nsamples)
vague(::Type{ SampleList }, dims::Tuple{Int, Int}; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES) = sample_list_vague(Matrixvariate, dims, nsamples)
vague(::Type{ SampleList }, dim1::Int, dim2::Int; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)  = sample_list_vague(Matrixvariate, (dim1, dim2), nsamples)

## prod related stuff

# `x` is proposal distribution
# `y` is integrand distribution
function approximate_prod_with_sample_list(x, y; nsamples = DEFAULT_SAMPLE_LIST_N_SAMPLES, rng = Random.GLOBAL_RNG)

    samples = rand(rng, x, nsamples)

    T            = promote_type(eltype(x), eltype(y))
    raw_weights  = similar(samples, T) # un-normalised
    norm_weights = similar(samples, T) # normalised

    H_x         = zero(T)
    weights_sum = zero(T)

    @turbo for i in 1:nsamples
        # Apply log-pdf functions to the samples
        log_sample_x = logpdf(x, samples[i])
        log_sample_y = logpdf(y, samples[i])

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
    logproposal  = x
    logintegrand = y

    meta = SampleListMeta(raw_weights, entropy, logproposal, logintegrand)

    return SampleList(samples, norm_weights, meta)
end

## Preallocation utilities

preallocate_samples(::Type{T}, dims::Tuple, length::Int) where T = Vector{T}(undef, length * prod(dims))

## Cache utilities

function sample_list_mean(::Type{ U }, sl::SampleList) where U
    cache = get_cache(sl)
    mean  = get_mean_storage(cache)
    if !is_mean_cached(cache)
        mean = cache_mean!(cache, sample_list_mean!(mean, U, sl))
    end
    return mean
end

function sample_list_mean_cov(::Type{ U }, sl::SampleList) where U
    cache = get_cache(sl)
    mean  = sample_list_mean(U, sl)
    cov   = get_cov_storage(cache)
    if !is_cov_cached(cache)
        cov = cache_cov!(cache, sample_list_covm!(cov, mean, U, sl))
    end
    return (mean, cov)
end

## Specific implementations

## Univariate

function sample_list_mean!(μ, ::Type{ Univariate }, sl::SampleList)
    n, samples, weights = get_data(sl)
    @turbo for i in 1:n
        μ += weights[i] * samples[i]
    end
    return μ
end

function sample_list_covm!(σ², μ, ::Type{ Univariate }, sl::SampleList)
    n, samples, weights = get_data(sl)
    @turbo for i in 1:n
        σ² += weights[i] * abs2(samples[i] - μ)
    end 
    σ² = (n / (n - 1)) * σ²
    return σ²
end 

function sample_list_mean_var(::Type{ Univariate }, sl::SampleList)
    return sample_list_mean_cov(Univariate, sl)
end 

function sample_list_logmean(::Type{ Univariate }, sl::SampleList)
    n, samples, weights = get_data(sl)
    logμ = sample_list_zero_element(sl)
    @turbo for i in 1:n
        logμ += weights[i] * log(samples[i])
    end
    return logμ
end

function sample_list_meanlogmean(::Type{ Univariate }, sl::SampleList)
    n, samples, weights = get_data(sl)
    μlogμ = sample_list_zero_element(sl)
    @turbo for i in 1:n
        μlogμ += weights[i] * samples[i] * log(samples[i])
    end
    return μlogμ
end

function sample_list_mirroredlogmean(::Type{ Univariate }, sl::SampleList)
    @assert all(0 .<= sl .< 1) "mirroredlogmean does not apply to variables outside of the range [0, 1]"
    n, samples, weights = get_data(sl)
    mirμ = sample_list_zero_element(sl)
    @turbo for i in 1:n
        mirμ += weights[i] * log(1 - samples[i])
    end
    return mirμ
end

function sample_list_vague(::Type{ Univariate }, nsamples::Int)
    targetdist   = vague(Uniform)
    preallocated = preallocate_samples(Float64, (), nsamples)
    rand!(targetdist, preallocated)
    return SampleList(Val(()), preallocated, OneDivNVector(Float64, nsamples), nothing)
end

## Multivariate

function sample_list_mean!(μ, ::Type{ Multivariate }, sl::SampleList) 
    n, samples, weights = get_data(sl)
    k = length(μ)
    @turbo for i in 1:n, j in 1:k
        μ[j] += (weights[i] * samples[(i - 1) * k + j])
    end
    return μ
end

function sample_list_covm!(Σ, μ, ::Type{ Multivariate }, sl::SampleList)
    n, samples, weights = get_data(sl)
    tmp = similar(μ)
    k   = length(tmp)
    
    @turbo for i in 1:n
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

function sample_list_mean_var(::Type{ Multivariate }, sl::SampleList)
    μ, Σ = sample_list_mean_cov(Multivariate, sl)
    return μ, diag(Σ)
end 

function sample_list_logmean(::Type{ Multivariate }, sl::SampleList)
    n, samples, weights = get_data(sl)
    logμ = sample_list_zero_element(sl)
    k = length(logμ)
    @turbo for i in 1:n, j in 1:k
        logμ[j] += (weights[i] * log(samples[(i - 1) * k + j]))
    end
    return logμ
end

function sample_list_meanlogmean(::Type{ Multivariate }, sl::SampleList)
    n, samples, weights = get_data(sl)
    μlogμ = sample_list_zero_element(sl)
    k = length(μlogμ)
    @turbo for i in 1:n, j in 1:k
        cs = samples[(i - 1) * k + j]
        μlogμ[j] += (weights[i] * cs * log(cs))
    end
    return μlogμ
end

function sample_list_vague(::Type{ Multivariate }, dims::Int, nsamples::Int)
    targetdist   = vague(Uniform)
    preallocated = preallocate_samples(Float64, (dims, ), nsamples)
    rand!(targetdist, preallocated)
    return SampleList(Val((dims, )), preallocated, OneDivNVector(Float64, nsamples), nothing)
end

## Matrixvariate

function sample_list_mean!(μ, ::Type{ Matrixvariate }, sl::SampleList) 
    n, samples, weights = get_data(sl)
    k = length(μ)
    @turbo for i in 1:n, j in 1:k
        μ[j] += (weights[i] * samples[(i - 1) * k + j])
    end
    return μ
end

function sample_list_covm!(Σ, μ, ::Type{ Matrixvariate }, sl::SampleList)
    n, samples, weights = get_data(sl)
    k   = length(μ)
    rμ  = reshape(μ, k)
    tmp = similar(rμ)
    @turbo for i in 1:n
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

function sample_list_mean_var(::Type{ Matrixvariate }, sl::SampleList)
    μ, Σ = sample_list_mean_cov(Matrixvariate, sl)
    return μ, reshape(diag(Σ), size(μ))
end

function sample_list_logmean(::Type{ Matrixvariate }, sl::SampleList)
    n, samples, weights = get_data(sl)
    logμ = sample_list_zero_element(sl)
    k = length(logμ)
    @turbo for i in 1:n, j in 1:k
        logμ[j] += (weights[i] * log(samples[(i - 1) * k + j]))
    end
    return logμ
end

function sample_list_meanlogmean(::Type{ Matrixvariate }, sl::SampleList)
    n, samples, weights = get_data(sl)
    μlogμ = sample_list_zero_element(sl)
    k = length(μlogμ)
    @turbo for i in 1:n, j in 1:k
        cs = samples[(i - 1) * k + j]
        μlogμ[j] += (weights[i] * cs * log(cs))
    end
    return μlogμ
end

function sample_list_vague(::Type{ Matrixvariate }, dims::Tuple{Int, Int}, nsamples::Int)
    targetdist   = vague(Uniform)
    preallocated = preallocate_samples(Float64, dims, nsamples)
    rand!(targetdist, preallocated)
    return SampleList(Val(dims), preallocated, OneDivNVector(Float64, nsamples), nothing)
end

## Array operations, broadcasting and mapping

Base.iterate(sl::SampleList)             = (sl[1], 2)
Base.iterate(sl::SampleList, state::Int) = state <= length(sl) ? (sl[state], state + 1) : nothing

@inline Base.getindex(sl::SampleList, i::Int) = sample_list_get_index(variate_form(sl), ndims(sl), sl, i)

@inline function sample_list_get_index(::Type{ Univariate }, ndims, sl, i)
    return (get_linear_samples(sl)[i], get_linear_weights(sl)[i])
end

@inline function sample_list_get_index(::Type{ Multivariate }, ndims, sl, i)
    samples = get_linear_samples(sl)
    left  = (i - 1) * ndims + 1
    right = left + ndims - 1
    # ndims is compile-time here
    return (SVector{ndims}(view(samples, left:right)), get_linear_weights(sl)[i])
end

@inline function sample_list_get_index(::Type{ Matrixvariate }, ndims, sl, i)
    p = prod(ndims)
    samples = get_linear_samples(sl)
    left  = (i - 1) * p + 1
    right = left + p - 1
    # ndims are compile-time here
    return (SMatrix{ ndims[1], ndims[2] }(reshape(view(samples, left:right), ndims)), get_linear_weights(sl)[i])
end

Base.map(f::Function, sl::SampleList) = Base.broadcasted(f, sl)

@inline samples_list_inplace_transform(::Type{ Univariate }, f::Function) = @inline (output, input) -> output[] = f(input)
@inline samples_list_inplace_transform(::Type{ Multivariate }, f::Function) = @inline (output, input) -> copyto!(output, f(input))
@inline samples_list_inplace_transform(::Type{ Matrixvariate }, f::Function) = @inline (output, input) -> copyto!(output, f(input))

function Base.broadcasted(f::Function, sl::SampleList) 
    return sample_list_broadcasted_inplace(variate_form(sl), samples_list_inplace_transform(variate_form(sl), f), sl)
end

function sample_list_broadcasted_inplace(::Type{ Univariate }, f!::Function, sl::SampleList)
    n, samples, weights = get_data(sl)
    bsamples = preallocate_samples(eltype(samples), (1, ), n)
    for i in 1:n
        f!(view(bsamples, i), samples[i])
    end
    return SampleList(Val(()), bsamples, weights)
end

function sample_list_broadcasted_inplace(::Type{ Multivariate }, f!::Function, sl::SampleList)
    n, samples, weights = get_data(sl)
    k = ndims(sl)
    bsamples = preallocate_samples(eltype(samples), (k, ), n)
    @views for i in 1:n
        left  = (i - 1) * k + 1
        right = left + k - 1
        output = SizedVector{k}(bsamples[left:right])
        input  = SVector{k}(samples[left:right])
        f!(output, input)
    end
    return SampleList(Val((k, )), bsamples, weights)
end

function sample_list_broadcasted_inplace(::Type{ Matrixvariate }, f!::Function, sl::SampleList)
    n, samples, weights = get_data(sl)
    k = ndims(sl)
    p = prod(k)
    bsamples = preallocate_samples(eltype(samples), k, n)
    @views for i in 1:n
        left  = (i - 1) * p + 1
        right = left + p - 1
        output = SizedMatrix{k[1],k[2]}(reshape(bsamples[left:right], k))
        input  = SMatrix{k[1],k[2]}(reshape(samples[left:right], k))
        f!(output, input)
    end
    return SampleList(Val(k), bsamples, weights)
end