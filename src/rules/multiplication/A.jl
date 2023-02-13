
@rule typeof(*)(:A, Marginalisation) (m_out::PointMass, m_in::PointMass, meta::Union{<:AbstractCorrection, Nothing}) = PointMass(mean(m_in) \ mean(m_out))

@rule typeof(*)(:A, Marginalisation) (m_out::GammaDistributionsFamily, m_in::PointMass{<:Real}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    return GammaShapeRate(shape(m_out), rate(m_out) * mean(m_in))
end

# if A is a matrix, then the result is multivariate
@rule typeof(*)(:A, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in::PointMass{<:AbstractMatrix}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_in)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, A' * W_out * A)
    return MvNormalWeightedMeanPrecision(A' * ξ_out, W)
end

# if A is a vector, then the result is univariate
# this rule links to the special case (AbstractVector * Univariate) for forward (:out) rule 
@rule typeof(*)(:A, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in::PointMass{<:AbstractVector}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_in)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, dot(A, W_out, A))
    return NormalWeightedMeanPrecision(dot(A, ξ_out), W)
end

# if A is a scalar, then the input is either univariate or multivariate
@rule typeof(*)(:A, Marginalisation) (m_out::F, m_in::PointMass{<:Real}, meta::Union{<:AbstractCorrection, Nothing}) where {F <: NormalDistributionsFamily} = begin
    A = mean(m_in)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, A^2 * W_out)
    return convert(promote_variate_type(F, NormalWeightedMeanPrecision), A * ξ_out, W)
end

# specialized versions for mean-covariance parameterization
@rule typeof(*)(:A, Marginalisation) (m_out::MvNormalMeanCovariance, m_in::PointMass{<:AbstractMatrix}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_in)
    μ_out, Σ_out = mean_cov(m_out)

    z = fastcholesky(Σ_out)
    tmp = A' / z
    W = correction!(meta, tmp * A)

    return MvNormalWeightedMeanPrecision(tmp * μ_out, W)
end

@rule typeof(*)(:A, Marginalisation) (m_out::MvNormalMeanCovariance, m_in::PointMass{<:AbstractVector}, meta::Union{<:AbstractCorrection, Nothing}) = begin
    A = mean(m_in)
    μ_out, Σ_out = mean_cov(m_out)

    z = fastcholesky(Σ_out)
    tmp = A' / z
    W = correction!(meta, dot(tmp, A))

    return NormalWeightedMeanPrecision(dot(tmp, μ_out), W)
end
