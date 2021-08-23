export SampleList, SampleListMeta

import Base: show, ndims, length, size, precision
import Distributions: mean, var, cov, std

using LoopVectorization

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
struct SampleList{S, W, M} 
    samples :: S
    weights :: W
    meta    :: M

    function SampleList(samples::S, weights::W, meta::M = nothing) where { S, W, M }
        @assert length(samples) === length(weights) "Invalid sample list samples and weights lengths. `samples` has length $(length(samples)), `weights` has length $(length(weights))"
        @assert eltype(weights) <: Number "Invalid eltype of weights container. Should be a subtype of `Number`, but $(eltype(weights)) has been found."
        return new{S, W, M}(samples, weights, meta)
    end
end

Base.show(io::IO, sl::SampleList) = print(io, "SampleList(", length(sl), ")")

SampleList(samples::S) where S = SampleList(samples, OneDivNVector(deep_eltype(S), length(samples)))

const DEFAULT_SAMPLE_LIST_N_SAMPLES = 5000

## Variate forms

variate_form(::SampleList{ S }) where S = sample_list_variate_form(eltype(S))

sample_list_variate_form(::Type{ T }) where { T <: Real }                         = Univariate
sample_list_variate_form(::Type{ V }) where { T <: Real, V <: AbstractVector{T} } = Multivariate
sample_list_variate_form(::Type{ M }) where { T <: Real, M <: AbstractMatrix{T} } = Matrixvariate

## Getters

get_samples(sl::SampleList) = sl.samples
get_weights(sl::SampleList) = sl.weights
get_meta(sl::SampleList)    = sample_list_check_meta(sl.meta)

sample_list_check_meta(meta::Any)     = meta
sample_list_check_meta(meta::Nothing) = error("SampleList object has not associated meta information with it.")

get_unnormalised_weights(sl::SampleList) = get_unnormalised_weights(get_meta(sl))
get_entropy(sl::SampleList)              = get_entropy(get_meta(sl))
get_logproposal(sl::SampleList)          = get_logproposal(get_meta(sl))
get_logintegrand(sl::SampleList)         = get_logintegrand(get_meta(sl))

call_logproposal(sl::SampleList, x)  = call_logproposal(get_logproposal(sl), x)
call_logintegrand(sl::SampleList, x) = call_logintegrand(get_logintegrand(sl), x)

Base.length(sl::SampleList)            = length(get_samples(sl))
Base.ndims(sl::SampleList)             = sample_list_ndims(variate_form(sl), sl)
Base.size(sl::SampleList)              = (length(sl), )

sample_list_ndims(::Type{ Univariate }, sl::SampleList)    = 1
sample_list_ndims(::Type{ Multivariate }, sl::SampleList)  = length(first(get_samples(sl)))
sample_list_ndims(::Type{ Matrixvariate }, sl::SampleList) = size(first(get_samples(sl)))

## Statistics 

# Returns a zeroed container for mean
sample_list_zero_element(sl::SampleList) = zero(first(get_weights(sl))) * zero(first(get_samples(sl)))

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

## Specific implementations

## Univariate

function sample_list_mean(::Type{ Univariate }, sl::SampleList)
    n = length(sl)
    μ = sample_list_zero_element(sl)
    weights = get_weights(sl)
    samples = get_samples(sl)
    @turbo for i in 1:n
        μ = μ + weights[i] * samples[i]
    end
    return μ
end

function sample_list_mean_cov(::Type{ Univariate }, sl::SampleList)
    n  = length(sl)
    μ  = mean(sl)
    σ² = sample_list_zero_element(sl)
    weights = get_weights(sl)
    samples = get_samples(sl)
    @turbo for i in 1:n
        σ² = σ² + weights[i] * abs2(samples[i] - μ)
    end 
    σ² = (n / (n - 1)) * σ²
    return μ, σ²
end 

function sample_list_mean_var(::Type{ Univariate }, sl::SampleList)
    return sample_list_mean_cov(Univariate, sl)
end 

function sample_list_logmean(::Type{ Univariate }, sl::SampleList)
    n = length(sl)
    logμ = sample_list_zero_element(sl)
    weights = get_weights(sl)
    samples = get_samples(sl)
    @turbo for i in 1:n
        logμ = logμ + weights[i] * log(samples[i])
    end
    return logμ
end

function sample_list_meanlogmean(::Type{ Univariate }, sl::SampleList)
    n = length(sl)
    μlogμ = sample_list_zero_element(sl)
    weights = get_weights(sl)
    samples = get_samples(sl)
    @turbo for i in 1:n
        μlogμ = μlogμ + weights[i] * samples[i] * log(samples[i])
    end
    return μlogμ
end

function sample_list_mirroredlogmean(::Type{ Univariate }, sl::SampleList)
    samples = get_samples(sl)
    @assert all(0 .<= sl .< 1) "mirroredlogmean does not apply to variables outside of the range [0, 1]"
    return mapreduce(z -> z[1] * log(1 - z[2]), +, zip(get_weights(sl), get_samples(sl)); init = sample_list_zero_element(sl))
end

function sample_list_vague(::Type{ Univariate }, length::Int)
    targetdist = vague(Uniform)
    return SampleList(rand(targetdist, length))
end

## Multivariate

function sample_list_mean(::Type{ Multivariate }, sl::SampleList) 
    n = length(sl)
    μ = sample_list_zero_element(sl)
    weights = get_weights(sl)
    samples = get_samples(sl)
    k = length(μ)
    for i in 1:n
        @turbo for j in 1:k
            μ[j] = μ[j] + (weights[i] * samples[i][j])
        end
    end
    return μ
end

function sample_list_mean_cov(::Type{ Multivariate }, sl::SampleList)
    n  = length(sl)
    μ  = mean(sl)

    Σ   = zeros(eltype(μ), length(μ), length(μ))

    weights = get_weights(sl)
    samples = get_samples(sl)

    tmp = similar(μ)
    k   = length(tmp)

    for i in 1:n 
        @turbo for j in 1:k
            tmp[j] = samples[i][j] - μ[j]
        end
        # Fast equivalent of Σ += w .* (tmp * tmp')
        # mul!(C, A, B, α, β) does C = A * B * α + C * β
        mul!(Σ, tmp, tmp', weights[i], 1)
    end

    return μ, Σ
end 

function sample_list_mean_var(::Type{ Multivariate }, sl::SampleList)
    μ, Σ = sample_list_mean_cov(Multivariate, sl)
    return μ, diag(Σ)
end 

function sample_list_logmean(::Type{ Multivariate }, sl::SampleList)
    n = length(sl)
    logμ = sample_list_zero_element(sl)
    weights = get_weights(sl)
    samples = get_samples(sl)
    k = length(logμ)
    for i in 1:n
        @turbo for j in 1:k
            logμ[j] = logμ[j] + weights[i] * log(samples[i][j])
        end
    end
    return logμ
end

function sample_list_meanlogmean(::Type{ Multivariate }, sl::SampleList)
    n = length(sl)
    μlogμ = sample_list_zero_element(sl)
    weights = get_weights(sl)
    samples = get_samples(sl)
    k = length(μlogμ)
    for i in 1:n
        @turbo for j in 1:k
            μlogμ[j] = μlogμ[j] + weights[i] * samples[i][j] * log(samples[i][j])
        end
    end
    return μlogμ
end

function sample_list_vague(::Type{ Multivariate }, dims::Int, length::Int)
    targetdist = vague(Uniform)
    return SampleList([ rand(targetdist, dims) for _ in 1:length ])
end

## Matrixvariate

function sample_list_mean(::Type{ Matrixvariate }, sl::SampleList) 
    n = length(sl)
    μ = sample_list_zero_element(sl)
    weights = get_weights(sl)
    samples = get_samples(sl)
    k = length(μ)
    for i in 1:n
        @turbo for j in 1:k
            μ[j] = μ[j] + (weights[i] * samples[i][j])
        end
    end
    return μ
end

function sample_list_mean_cov(::Type{ Matrixvariate }, sl::SampleList)
    n  = length(sl)
    μ  = mean(sl)
    weights = get_weights(sl)
    samples = get_samples(sl)
    k   = length(μ)
    rμ  = reshape(μ, k)
    tmp = similar(rμ)
    Σ   = zeros(eltype(rμ), length(rμ), length(rμ))

    for i in 1:n
        @turbo for j in 1:k
            tmp[j] = samples[i][j] - μ[j]
        end
        # Fast equivalent of Σ += w .* (tmp * tmp')
        # mul!(C, A, B, α, β) does C = A * B * α + C * β
        mul!(Σ, tmp, tmp', weights[i], 1)
    end

    return μ, Σ
end 

function sample_list_mean_var(::Type{ Matrixvariate }, sl::SampleList)
    μ, Σ = sample_list_mean_cov(Matrixvariate, sl)
    return μ, reshape(diag(Σ), size(μ))
end

function sample_list_logmean(::Type{ Matrixvariate }, sl::SampleList)
    n = length(sl)
    logμ = sample_list_zero_element(sl)
    weights = get_weights(sl)
    samples = get_samples(sl)
    k = length(logμ)
    for i in 1:n
        @turbo for j in 1:k
            logμ[j] = logμ[j] + weights[i] * log(samples[i][j])
        end
    end
    return logμ
end

function sample_list_meanlogmean(::Type{ Matrixvariate }, sl::SampleList)
    n = length(sl)
    μlogμ = sample_list_zero_element(sl)
    weights = get_weights(sl)
    samples = get_samples(sl)
    k = length(μlogμ)
    for i in 1:n
        @turbo for j in 1:k
            μlogμ[j] = μlogμ[j] + weights[i] * samples[i][j] * log(samples[i][j])
        end
    end
    return μlogμ
end

function sample_list_vague(::Type{ Matrixvariate }, dims::Tuple{Int, Int}, length::Int)
    targetdist = vague(Uniform)
    return SampleList([ rand(targetdist, dims[1], dims[2]) for _ in 1:length ])
end