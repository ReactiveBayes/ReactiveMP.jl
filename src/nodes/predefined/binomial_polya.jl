export BinomialPolya, BinomialPolyaMeta

"""
    BinomialPolya

A node type representing a Binomial likelihood with linear predictor through logistic. A Normal prior on the weights is used. 
The prior is augmented with a PolyaGamma distribution, which is used for modeling count data with overdispersion. 
This implementation follows the PolyaGamma augmentation scheme for Bayesian inference. Can be used for Binomial regression. 
"""
struct BinomialPolya end

"""
    BinomialPolyaMeta

Metadata structure for the BinomialPolya node. It will be passed to rules. In case no meta is provided,
the rules will use the means to compute the messages. Both schemes yield very similar results.

# Fields
- `n_samples::Int`: Number of samples to use for Monte Carlo estimation of the average energy.
                    Default is 1, as increasing it adds computational cost without significant benefit.
- `rng::AbstractRNG`: Random number generator to use for sampling. Defaults to Random.default_rng().
"""
struct BinomialPolyaMeta
    n_samples::Int
    rng::AbstractRNG
end

# Constructor with default RNG
BinomialPolyaMeta(n_samples::Int = 1) = BinomialPolyaMeta(n_samples, Random.default_rng())

getn_samples(meta::BinomialPolyaMeta) = meta.n_samples
default_meta(::Type{BinomialPolya}) = nothing

@node BinomialPolya Stochastic [y, x, n, β]

@average_energy BinomialPolya (q_y::PointMass, q_x::PointMass, q_n::PointMass, q_β::Any, meta::Union{BinomialPolyaMeta, Nothing}) = begin
    y = mean(q_y)
    x = mean(q_x)
    n = mean(q_n)
    β = mean(q_β)

    if meta === nothing
        term1 = -n * log((1 + exp(-dot(x, β))))
    else
        n_samples = getn_samples(meta)
        βsamples = rand(meta.rng, q_β, n_samples)
        term1_vec = map(βsample -> (-n * log((1 + exp(-dot(x, βsample))))), eachcol(βsamples))
        term1 = mean(term1_vec)
    end

    term1 = -n * log((1 + exp(-dot(x, β))))
    term2 = (y - n) * dot(x, β)
    term3 = loggamma(n + 1) - (loggamma(n - y + 1) + loggamma(y + 1))

    return -(term1 + term2 + term3)
end
