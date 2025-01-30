export MultinomialPolya, MultinomialPolyaMeta

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
@average_energy MultinomialPolya (q_x::PointMass, q_N::PointMass, q_ψ::Any, meta::Union{MultinomialPolyaMeta, Nothing}) = begin
    return 0.0
end