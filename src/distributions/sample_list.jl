export SampleList

import Base: show
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

## Statistics 

# Distributions.mean

## 

vague(::Type{ SampleList }; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)                        = _vague_sample_list(Univariate, nsamples)
vague(::Type{ SampleList }, dims::Int; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)             = _vague_sample_list(Multivariate, ndims, nsamples)
vague(::Type{ SampleList }, dims::Tuple{Int, Int}; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES) = _vague_sample_list(Matrixvariate, ndims, nsamples)

# function _vague_sample_list(::Univariate, length::Int)
#     return SampleList(rand(length))
# end

# function _vague_sample_list(::Multivariate, dims::Int, length::Int)
#     return SampleList([ rand(dims) for _ in 1:length ])
# end

# function _vague_sample_list(::Matrixvariate, dims::Tuple{Int, Int}, length::Int)
#     return SampleList([ randn(dims...) for _ in 1:length ])
# end
