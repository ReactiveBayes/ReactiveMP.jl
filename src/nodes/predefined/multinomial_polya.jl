const MULTINOMIAL_POLYA_CUBATURE_POINTS = 21
export MultinomialPolya, MultinomialPolyaMeta, logistic_stick_breaking, inverse_logistic_stick_breaking, compose_Nks
using PolyaGammaHybridSamplers
"""
    MultinomialPolya

A node type representing a MultinomialPolya likelihood with linear predictor through softmax. A Normal prior on the weights is used. 
The prior is augmented with a PolyaGamma distribution, which is used for modeling count data with overdispersion. 
This implementation follows the PolyaGamma augmentation scheme for Bayesian inference. Can be used for Multinomial regression. 
"""
struct MultinomialPolya end

"""
    MultinomialPolyaMeta

Metadata structure for the MultinomialPolya node. It will be passed to rules. In case no meta is provided,
the rules will use the means to compute the messages. Both schemes yield very similar results.

# Fields
- `n_samples::Int`: Number of samples to use for Monte Carlo estimation of the average energy.
                    Default is 1, as increasing it adds computational cost without significant benefit.
- `rng::AbstractRNG`: Random number generator to use for sampling. Defaults to Random.default_rng().
"""
struct MultinomialPolyaMeta
    n_samples::Int
    rng::AbstractRNG
end

# Constructor with default RNG
MultinomialPolyaMeta(n_samples::Int = 1) = MultinomialPolyaMeta(n_samples, Random.default_rng())

getn_samples(meta::MultinomialPolyaMeta) = meta.n_samples
default_meta(::Type{MultinomialPolya}) = nothing

@node MultinomialPolya Stochastic [x, N, ψ]

@average_energy MultinomialPolya (q_x::Any, q_N::PointMass, q_ψ::Union{GaussianDistributionsFamily, PointMass}, meta::Union{MultinomialPolyaMeta, Nothing}) = begin
    N             = mean(q_N)
    K             = first(size(mean(q_x)))
    x             = mean(q_x)
    T             = promote_samplefloattype(q_x, q_N, q_ψ)
    μ_ψ           = mean(q_ψ)
    v_ψ           = var(q_ψ)
    Nks           = compose_Nks(x, N)
    method        = ReactiveMP.ghcubature(MULTINOMIAL_POLYA_CUBATURE_POINTS)
    weights(m, v) = ReactiveMP.getweights(method, m, v)
    points(m, v)  = ReactiveMP.getpoints(method, m, v)
    expectations  = map((m, v) -> mapreduce((w, p) -> w * softplus(p), +, weights(m, v), points(m, v)), μ_ψ, v_ψ)

    if q_x isa PointMass
        term1 = -mapreduce((Nk, y) -> loggamma(Nk + 1) - loggamma(Nk - y + 1) - loggamma(y + 1), +, Nks, x)
    elseif q_x isa Multinomial || q_x isa Categorical
        if N != 1
            p = q_x.p
            binomials = map(x -> Binomial(N, x), p)
            term1 = -sum(expected_log_gamma.(binomials)) + log(N)
        else
            term1 = 0
        end
    else
        error("Unsupported distribution for x: $(typeof(q_x))")
    end
    term2 = -sum(x[1:(K - 1)] .* μ_ψ)
    term3 = mapreduce((e, Nk) -> e * Nk, +, expectations, Nks)
    return term1 + term2 + term3
end

function expected_log_gamma(binom)
    return mapreduce((p) -> loggamma(p + 1) * pdf(binom, p), +, collect(0:(binom.n)))
end

function logistic_stick_breaking(m)
    Km1 = length(m)

    p = Array{Float64}(undef, Km1 + 1)
    remaining = 1.0
    @inbounds for i in 1:Km1
        v = logistic(m[i])
        p[i] = v * remaining
        remaining *= (1 - v)
    end
    p[end] = remaining
    return p
end

function compose_Nks(x, N)
    T = eltype(x)
    K = length(x)
    Nks = Vector{T}(undef, K - 1)
    prev_sum = zero(T)
    @inbounds for k in 1:(K - 1)
        Nks[k] = N - prev_sum
        if k < K - 1
            prev_sum += x[k]
        end
    end
    return Nks
end
