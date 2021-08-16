export SampleList

import Distributions: mean, var, cov, std

struct SampleList{ S, W } 
    samples :: S
    weights :: W

    function SampleList{S, W}(samples::S, weights::W) where { S, W }
        if length(samples) !== length(weights)
            sample_list_throw_invalid_lengths(samples, weights)
        elseif eltype(weights) !<: Number
            sample_list_throw_invalid_weights_eltype(weights)  
        end
        return new(samples, weights)
    end
end

## Constructors

# https://github.com/JuliaLang/julia/issues/29688
sample_list_throw_invalid_lengths(samples, weights) = error("Invalid sample list samples and weights lengths. `samples` has length $(length(samples)), `weights` has length $(length(weights))")
sample_list_throw_invalid_weights_eltype(weights)   = error("Invalid eltype of weights container. Should be a subtype of `Number`, but $(eltype(weigths)) has been found.")

## Variate forms

variate_form(::SampleList{ S }) where S = _sample_list_variate_form(eltype(S))

_sample_list_variate_form(::Type{ T }) where { T <: Real }                         = Univariate
_sample_list_variate_form(::Type{ V }) where { T <: Real, V <: AbstractVector{T} } = Multivariate
_sample_list_variate_form(::Type{ M }) where { T <: Real, M <: AbstractMatrix{T} } = Matrixvariate

## Getters

getsamples(sl::SampleList) = sl.samples
getweights(sl::SampleList) = sl.weights

##

