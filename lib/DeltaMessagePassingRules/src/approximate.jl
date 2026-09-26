"""
    approximate_normal(method, f, distributions::Tuple)

The normal approximation of `f(x₁, …, xₙ)` for independent normal inputs `distributions`, by
`method`, [`Unscented`](@extref MessagePassingRulesApproximations.Unscented) or
[`Linearization`](@extref MessagePassingRulesApproximations.Linearization): a normal with the
pushforward's mean and covariance, univariate or multivariate as the output of `f` is. The
message rules towards `out`, and towards an input through a known inverse, are this call.
"""
function approximate_normal end

function approximate_normal(method::Unscented, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}
    statistics = mean_cov.(distributions)
    μ, Σ = MessagePassingRulesApproximations.approximate(method, f, first.(statistics), last.(statistics))
    return convert(promote_variate_type(typeof(μ), NormalMeanVariance), μ, Σ)
end

# The inputs' joint pushed through the linear map of `f` at their means.
function approximate_normal(method::Linearization, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}
    statistics = mean_cov.(distributions)
    means, covs = first.(statistics), last.(statistics)
    μ_in, Σ_in = mean_cov(convert(JointNormal, means, covs))
    μ, Σ, _ = forward_statistics(method, f, means, covs, μ_in, Σ_in)
    return convert(promote_variate_type(typeof(μ), NormalMeanVariance), μ, Σ)
end

"""
    forward_statistics(method, f, μs, Σs, μ_in, Σ_in) -> (μ, Σ, C)

The mean `μ` and covariance `Σ` of `f` of the inputs, and the cross-covariance `C` of the inputs
with it, by `method`, for inputs with means `μs` and covariances `Σs` (tuples, one entry per
input) whose joint has mean `μ_in` and covariance `Σ_in`. The rule for the joint over the inputs
smooths these with the message from `out`, by
[`smoothRTS`](@extref MessagePassingRulesApproximations.smoothRTS).
"""
function forward_statistics end

forward_statistics(method::Unscented, f::F, μs, Σs, μ_in, Σ_in) where {F} = unscented_statistics(method, f, μs, Σs)

function forward_statistics(method::Linearization, f::F, μs, Σs, μ_in, Σ_in) where {F}
    A, b = MessagePassingRulesApproximations.approximate(method, f, μs)
    return A * μ_in + b, A * Σ_in * A', Σ_in * A'
end
