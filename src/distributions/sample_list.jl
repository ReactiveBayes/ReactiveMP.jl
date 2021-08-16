export SampleList

import Base: show, ndims, length, size
import Distributions: mean, var, cov, std

"""
    SampleList

Generic distribution represented as a list of weighted samples.

# Arguments 
- `samples::S`
- `weights::W`: optional, equivalent to `fill(1 / N, N)` by default, where `N` is the length of `samples` container
"""
struct SampleList{S, W} 
    samples :: S
    weights :: W
    
    function SampleList{S, W}(samples::S, weights::W) where { S, W }
        return new(samples, weights)
    end
end

function SampleList(samples::S, weights::W) where { S, W }
    @assert length(samples) === length(weights) "Invalid sample list samples and weights lengths. `samples` has length $(length(samples)), `weights` has length $(length(weights))"
    @assert eltype(weights) <: Number "Invalid eltype of weights container. Should be a subtype of `Number`, but $(eltype(weights)) has been found."
    return SampleList{S, W}(samples, weights)
end

function SampleList(samples::S) where S
    weights = OneDivNVector(deep_eltype(S), length(samples))
    W       = typeof(weights)
    return SampleList{S, W}(samples, weights)
end 

const DEFAULT_SAMPLE_LIST_N_SAMPLES = 5000

## Variate forms

variate_form(::SampleList{ S }) where S = _sample_list_variate_form(eltype(S))

_sample_list_variate_form(::Type{ T }) where { T <: Real }                         = Univariate
_sample_list_variate_form(::Type{ V }) where { T <: Real, V <: AbstractVector{T} } = Multivariate
_sample_list_variate_form(::Type{ M }) where { T <: Real, M <: AbstractMatrix{T} } = Matrixvariate

## Getters

getsamples(sl::SampleList) = sl.samples
getweights(sl::SampleList) = sl.weights

Base.length(sl::SampleList)            = length(getsamples(sl))
Base.ndims(sl::SampleList)             = _sample_list_ndims(variate_form(sl), sl)
Base.size(sl::SampleList)              = (length(sl), )

_sample_list_ndims(::Type{ Univariate }, sl::SampleList)    = 1
_sample_list_ndims(::Type{ Multivariate }, sl::SampleList)  = length(first(getsamples(sl)))
_sample_list_ndims(::Type{ Matrixvariate }, sl::SampleList) = size(first(getsamples(sl)))

## Statistics 

_sample_list_zero(sl::SampleList) = zero(first(getweights(sl))) * zero(first(getsamples(sl)))

Distributions.mean(sl::SampleList) = mapreduce(z -> z[1] .* z[2], +, zip(getweights(sl), getsamples(sl)); init = _sample_list_zero(sl))

logmean(sl::SampleList)     = mapreduce(z -> z[1] .* log.(z[2]), +, zip(getweights(sl), getsamples(sl)); init = _sample_list_zero(sl))
meanlogmean(sl::SampleList) = mapreduce(z -> z[1] .* z[2] .* log.(z[2]), +, zip(getweights(sl), getsamples(sl)); init = _sample_list_zero(sl))

## 

vague(::Type{ SampleList }; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)                        = _vague_sample_list(Univariate, nsamples)
vague(::Type{ SampleList }, dims::Int; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)             = _vague_sample_list(Multivariate, dims, nsamples)
vague(::Type{ SampleList }, dims::Tuple{Int, Int}; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES) = _vague_sample_list(Matrixvariate, dims, nsamples)
vague(::Type{ SampleList }, dim1::Int, dim2::Int; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES) = _vague_sample_list(Matrixvariate, (dim1, dim2), nsamples)

function _vague_sample_list(::Type{ Univariate }, length::Int)
    targetdist = vague(Uniform)
    return SampleList(rand(targetdist, length))
end

function _vague_sample_list(::Type{ Multivariate }, dims::Int, length::Int)
    targetdist = vague(Uniform)
    return SampleList([ rand(targetdist, dims) for _ in 1:length ])
end

function _vague_sample_list(::Type{ Matrixvariate }, dims::Tuple{Int, Int}, length::Int)
    targetdist = vague(Uniform)
    return SampleList([ rand(targetdist, dims...) for _ in 1:length ])
end
