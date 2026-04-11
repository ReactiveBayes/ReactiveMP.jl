export ImportanceSamplingApproximation

using Random
using StatsBase

"""
    ImportanceSamplingApproximation <: AbstractApproximationMethod

Importance-sampling approximation for computing nonlinear expectations. Draws
samples from a proposal distribution and reweights them to estimate the target
expectation. It is the most flexible approximation method — it makes no
assumptions about the shape of the factor — but it converges slowly in high
dimensions.

# Fields
- `rng`: random number generator
- `nsamples`: number of samples used per approximation call
- `resampling`: if `true`, resamples when the effective sample size drops below `N/10`
"""
struct ImportanceSamplingApproximation{T, R}
    rng        :: R
    nsamples   :: Int
    bsamples   :: Vector{T}
    bweights   :: Vector{T}
    resampling :: Bool
    rsamples   :: Vector{T}
end

function ImportanceSamplingApproximation(
    rng::R, nsamples::Int; resampling::Bool = true
) where {R}
    return ImportanceSamplingApproximation(
        Float64, rng, nsamples; resampling = resampling
    )
end

function ImportanceSamplingApproximation(
    ::Type{T}, rng::R, nsamples::Int; resampling::Bool = true
) where {T, R}
    return ImportanceSamplingApproximation{T, R}(
        rng,
        nsamples,
        Vector{T}(undef, nsamples),
        Vector{T}(undef, nsamples),
        resampling,
        Vector{T}(undef, nsamples),
    )
end

getsamples(approximation::ImportanceSamplingApproximation, distribution)           = getsamples(approximation, distribution, approximation.nsamples)
getsamples(approximation::ImportanceSamplingApproximation, distribution, nsamples) = rand(approximation.rng, distribution, nsamples)

function approximate_meancov(
    approximation::ImportanceSamplingApproximation, g::Function, distribution
)

    # We use preallocated arrays to sample and compute transformed samples and weightd
    rand!(approximation.rng, distribution, approximation.bsamples)
    map!(g, approximation.bweights, approximation.bsamples)

    if approximation.resampling
        n_eff = 1 / sum(Base.Generator(abs2, approximation.bweights))

        # Here we assume that lengths of bweights ans bsamples vectors are the same
        N = length(approximation.bweights)

        if n_eff < N / 10
            # We use rsamples as a temporary buffer here
            copyto!(approximation.rsamples, 1, approximation.bsamples, 1, N)
            # Here rsamples are equal to bsamples, but during sampling bsamples will be overwritten
            sample!(
                approximation.rng,
                approximation.rsamples,
                Weights(approximation.bweights),
                approximation.bsamples,
            )
            fill!(approximation.bweights, 1 / N)
        end
    end

    normalization = sum(approximation.bweights)

    if iszero(normalization)
        return mean(distribution), var(distribution)
    end

    map!(
        Base.Fix2(/, normalization),
        approximation.bweights,
        approximation.bweights,
    )

    m = mapreduce(prod, +, zip(approximation.bweights, approximation.bsamples))

    _v = let m = m
        (r) -> r[1] * (r[2] - m)^2
    end

    v = mapreduce(_v, +, zip(approximation.bweights, approximation.bsamples))

    if isnan(m) || isnan(v) || isinf(m) || iszero(v)
        return mean(distribution), var(distribution)
    end

    return m, v
end
