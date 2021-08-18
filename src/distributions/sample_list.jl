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

variate_form(::SampleList{ S }) where S = sample_list_variate_form(eltype(S))

sample_list_variate_form(::Type{ T }) where { T <: Real }                         = Univariate
sample_list_variate_form(::Type{ V }) where { T <: Real, V <: AbstractVector{T} } = Multivariate
sample_list_variate_form(::Type{ M }) where { T <: Real, M <: AbstractMatrix{T} } = Matrixvariate

## Getters

getsamples(sl::SampleList) = sl.samples
getweights(sl::SampleList) = sl.weights

Base.length(sl::SampleList)            = length(getsamples(sl))
Base.ndims(sl::SampleList)             = sample_list_ndims(variate_form(sl), sl)
Base.size(sl::SampleList)              = (length(sl), )

sample_list_ndims(::Type{ Univariate }, sl::SampleList)    = 1
sample_list_ndims(::Type{ Multivariate }, sl::SampleList)  = length(first(getsamples(sl)))
sample_list_ndims(::Type{ Matrixvariate }, sl::SampleList) = size(first(getsamples(sl)))

## Statistics 

# Returns a zeroed container for mean vector
sample_list_zero_element(sl::SampleList) = zero(first(getweights(sl))) * zero(first(getsamples(sl)))

# Generic mean_cov

mean_cov(sl::SampleList) = sample_list_mean_cov(variate_form(sl), sl)
mean_var(sl::SampleList) = sample_list_mean_var(variate_form(sl), sl)

function sample_list_mean_cov(::Type{ Univariate }, sl::SampleList)
    n  = length(sl)
    μ  = mean(sl)
    σ² = (n / (n - 1)) * mapreduce((z) -> z[1] * abs2(z[2] - μ), +, zip(getweights(sl), getsamples(sl)); init = sample_list_zero_element(sl))
    return μ, σ²
end 

function sample_list_mean_cov(::Type{ Multivariate }, sl::SampleList)
    n  = length(sl)
    μ  = mean(sl)

    Σ   = zeros(eltype(μ), length(μ), length(μ))

    weights = getweights(sl)
    samples = getsamples(sl)

    tmp = similar(μ)
    k   = length(tmp)

    for i in 1:n 
        for j in 1:k
            tmp[j] = samples[i][j] - μ[j]
        end
        # Fast equivalent of Σ += w .* (tmp * tmp')
        # mul!(C, A, B, α, β) does C = A * B * α + C * β
        mul!(Σ, tmp, tmp', weights[i], 1)
    end

    return μ, Σ
end 

function sample_list_mean_cov(::Type{ Matrixvariate }, sl::SampleList)
    error("sample_list_mean_cov for Matrixvariate distribution is broken")

    n  = length(sl)
    μ  = mean(sl)

    cov1 = zeros(eltype(μ), first(ndims(sl)), last(ndims(sl)))
    cov2 = zeros(eltype(μ), first(ndims(sl)), last(ndims(sl)))

    weights = getweights(sl)
    samples = getsamples(sl)

    tmp = similar(μ)
    k   = length(tmp)

    for i in 1:n
        w = weights[i]

        for j in 1:k
            tmp[j] = samples[i][j] - μ[j]
        end

        # Fast equivalent of cov1 += w .* (tmp * tmp')
        # mul!(C, A, B, α, β) does C = A * B * α + C * β
        mul!(cov1, tmp, tmp', w, 1)
        mul!(cov2, tmp', tmp, w, 1)
    end

    S = n / (n - 1)

    cov1 .*= S
    cov2 .*= S

    Σ = kron(cov1, cov2)

    return μ, Σ
end 

function sample_list_mean_var(::Type{ Univariate }, sl::SampleList)
    return sample_list_mean_cov(Univariate, sl)
end 

function sample_list_mean_var(::Type{ Multivariate }, sl::SampleList)
    μ, Σ = sample_list_mean_cov(Multivariate, sl)
    return μ, diag(Σ)
end 

function sample_list_mean_var(::Type{ Matrixvariate }, sl::SampleList)
    μ, W = sample_list_mean_cov(Matrixvariate, sl)
    n    = isqrt(length(μ))
    return μ, reshape(diag(W), n, n)
end 

##

Distributions.mean(sl::SampleList)      = mapreduce(z -> z[1] .* z[2], +, zip(getweights(sl), getsamples(sl)); init = sample_list_zero_element(sl))
Distributions.var(sl::SampleList)       = last(mean_var(sl))
Distributions.cov(sl::SampleList)       = last(mean_cov(sl))
Distributions.invcov(sl::SampleList)    = cholinv(cov(sl))
Distributions.std(sl::SampleList)       = cholsqrt(cov(sl))
Distributions.logdetcov(sl::SampleList) = logdet(cov(sl))

logmean(sl::SampleList)     = mapreduce(z -> z[1] .* log.(z[2]), +, zip(getweights(sl), getsamples(sl)); init = sample_list_zero_element(sl))
meanlogmean(sl::SampleList) = mapreduce(z -> z[1] .* z[2] .* log.(z[2]), +, zip(getweights(sl), getsamples(sl)); init = sample_list_zero_element(sl))

mirroredlogmean(sl::SampleList) = sample_list_mirroredlogmean(variate_form(sl), sl)

function sample_list_mirroredlogmean(::Type{ Univariate }, sl::SampleList)
    samples = getsamples(sl)
    @assert all(0 .<= sl .< 1) "mirroredlogmean does not apply to variables outside of the range [0, 1]"
    return mapreduce(z -> z[1] * log(1 - z[2]), +, zip(getweights(sl), getsamples(sl)); init = sample_list_zero_element(sl))
end

## 

vague(::Type{ SampleList }; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)                        = sample_list_vague(Univariate, nsamples)
vague(::Type{ SampleList }, dims::Int; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)             = sample_list_vague(Multivariate, dims, nsamples)
vague(::Type{ SampleList }, dims::Tuple{Int, Int}; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES) = sample_list_vague(Matrixvariate, dims, nsamples)
vague(::Type{ SampleList }, dim1::Int, dim2::Int; nsamples::Int = DEFAULT_SAMPLE_LIST_N_SAMPLES)  = sample_list_vague(Matrixvariate, (dim1, dim2), nsamples)

function sample_list_vague(::Type{ Univariate }, length::Int)
    targetdist = vague(Uniform)
    return SampleList(rand(targetdist, length))
end

function sample_list_vague(::Type{ Multivariate }, dims::Int, length::Int)
    targetdist = vague(Uniform)
    return SampleList([ rand(targetdist, dims) for _ in 1:length ])
end

function sample_list_vague(::Type{ Matrixvariate }, dims::Tuple{Int, Int}, length::Int)
    targetdist = vague(Uniform)
    return SampleList([ rand(targetdist, dims[1], dims[2]) for _ in 1:length ])
end
