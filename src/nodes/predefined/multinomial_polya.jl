export MultinomialPolya, MultinomialPolyaMeta, logistic_stic_breaking
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

##TODO: Implement the average energy for MultinomialPolya
@average_energy MultinomialPolya (q_x::Union{PointMass, Multinomial}, q_N::PointMass, q_ψ::Union{GaussianDistributionsFamily,PointMass}, meta::Union{MultinomialPolyaMeta, Nothing}) = begin
    # Get parameters from variational distributions
    N = mean(q_N)
    K = length(mean(q_x))
    p = q_x isa PointMass ? mean(q_x) ./ N : probs(q_x)
    
    # Get ψ statistics
    μ_ψ = mean(q_ψ)
    Σ_ψ = cov(q_ψ)
    var_ψ = diag(Σ_ψ)
    
    # Compute expectations analytically
    linear_term = sum((N * p[k] - N/2) * μ_ψ[k] for k in 1:(K-1)) + (N * p[K] - N/2) * 0.0
    
    # E[ω_kψ_k^2] = E[ψ_k^2] * E[ω_k] using law of total expectation
    # For Polya-Gamma: E[ω_k] = N/(2c) * tanh(c/2) where c = |ψ_k|
    # We approximate E[tanh(|ψ_k|/2)/|ψ_k|] using Taylor expansion
    quadratic_term = sum(N/4 * (var_ψ[k] + μ_ψ[k]^2 - abs(μ_ψ[k])) for k in 1:(K-1))
    
    return linear_term - quadratic_term
end

function logistic_stic_breaking(m)
    Km1 = length(m)

    p = Array{Float64}(undef, Km1+1)
    p[1] = logistic(m[1])
    for i in 2:Km1
        p[i] = logistic(m[i])*(1 - sum(p[1:i-1]))
    end
    p[end] = 1 - sum(p[1:end-1])
    return p
end