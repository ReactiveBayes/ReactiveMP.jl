"""
    approximate_normal(method, f, distributions)

The normal approximation of `f(x₁, …, xₙ)` for independent normal inputs, through `method`:
a `NormalMeanVariance`, or its multivariate form, of the pushforward's mean and covariance.
"""
function approximate_normal(method::Unscented, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}
    statistics = mean_cov.(distributions)
    μ, Σ = MessagePassingRulesApproximations.approximate(method, f, first.(statistics), last.(statistics))
    return convert(promote_variate_type(typeof(μ), NormalMeanVariance), μ, Σ)
end
