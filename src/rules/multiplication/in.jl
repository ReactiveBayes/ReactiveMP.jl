
@rule typeof(*)(:in, Marginalisation) (m_out::PointMass, m_A::PointMass, meta::Union{<:AbstractCorrectionStrategy, Nothing}) = PointMass(mean(m_A) \ mean(m_out))

@rule typeof(*)(:in, Marginalisation) (m_out::GammaDistributionsFamily, m_A::PointMass{<:Real}, meta::Union{<:AbstractCorrectionStrategy, Nothing}) = begin
    return GammaShapeRate(shape(m_out), rate(m_out) * mean(m_A))
end

# if A is a matrix, then the result is multivariate
@rule typeof(*)(:in, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_A::PointMass{<:AbstractMatrix}, meta::Union{<:AbstractCorrectionStrategy, Nothing}) = begin
    A = mean(m_A)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, A' * W_out * A)
    return MvNormalWeightedMeanPrecision(A' * ξ_out, W)
end

# if A is a vector, then the result is univariate
# this rule links to the special case (AbstractVector * Univariate) for forward (:out) rule 
@rule typeof(*)(:in, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_A::PointMass{<:AbstractVector}, meta::Union{<:AbstractCorrectionStrategy, Nothing}) = begin
    A = mean(m_A)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, dot(A, W_out, A))
    return NormalWeightedMeanPrecision(dot(A, ξ_out), W)
end

# if A is a scalar, then the input is either univariate or multivariate
@rule typeof(*)(:in, Marginalisation) (m_out::F, m_A::PointMass{<:Real}, meta::Union{<:AbstractCorrectionStrategy, Nothing}) where {F <: NormalDistributionsFamily} = begin
    A = mean(m_A)
    @logscale -logdet(A)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, A^2 * W_out)
    return convert(promote_variate_type(F, NormalWeightedMeanPrecision), A * ξ_out, W)
end

# specialized versions for mean-covariance parameterization
@rule typeof(*)(:in, Marginalisation) (m_out::MvNormalMeanCovariance, m_A::PointMass{<:AbstractMatrix}, meta::Union{<:AbstractCorrectionStrategy, Nothing}) = begin
    A = mean(m_A)
    μ_out, Σ_out = mean_cov(m_out)

    z = fastcholesky(Σ_out)
    tmp = A' / z
    W = correction!(meta, tmp * A)

    return MvNormalWeightedMeanPrecision(tmp * μ_out, W)
end

@rule typeof(*)(:in, Marginalisation) (m_out::MvNormalMeanCovariance, m_A::PointMass{<:AbstractVector}, meta::Union{<:AbstractCorrectionStrategy, Nothing}) = begin
    A = mean(m_A)
    μ_out, Σ_out = mean_cov(m_out)

    z = fastcholesky(Σ_out)
    tmp = A' / z
    W = correction!(meta, dot(tmp, A))

    return NormalWeightedMeanPrecision(dot(tmp, μ_out), W)
end

@rule typeof(*)(:in, Marginalisation) (
    m_out::UnivariateGaussianDistributionsFamily, m_A::UnivariateGaussianDistributionsFamily, meta::Union{<:AbstractCorrectionStrategy, Nothing}
) = begin
    μ_A, var_A = mean_var(m_A)
    μ_out, var_out = mean_var(m_out)
    log_backwardpass = (x) -> -log(abs(x)) - 0.5 * log(2π * (var_A + var_out / x^2)) - 1 / 2 * (μ_out - x * μ_A)^2 / (var_A * x^2 + var_out)
    return ContinuousUnivariateLogPdf(log_backwardpass)
end

@rule typeof(*)(:in, Marginalisation) (m_out::UnivariateDistribution, m_A::UnivariateDistribution, meta::Union{<:AbstractCorrectionStrategy, Nothing}) = begin
    return @call_rule typeof(*)(:A, Marginalisation) (m_out = m_out, m_in = m_A, meta = meta)
end
