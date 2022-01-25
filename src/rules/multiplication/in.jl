
@rule typeof(*)(:in, Marginalisation) (m_out::PointMass, m_A::PointMass) = PointMass(inv(m_A) * mean(m_out))

@rule typeof(*)(:in, Marginalisation) (m_out::GammaDistributionsFamily, m_A::PointMass{ <: Real }) = begin 
    return GammaShapeRate(shape(m_out), rate(m_out) * mean(m_A))
end

# if A is a matrix, then the result is multivariate
@rule typeof(*)(:in, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_A::PointMass{ <: AbstractMatrix }, meta::AbstractCorrection) = begin
    A = mean(m_A)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, A' * W_out * A)
    return MvNormalWeightedMeanPrecision(A' * ξ_out, W)
end

# if A is a vector, then the result is univariate
# this rule links to the special case (AbstractVector * Univariate) for forward (:out) rule 
@rule typeof(*)(:in, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_A::PointMass{ <: AbstractVector }, meta::AbstractCorrection) = begin
    A = mean(m_A)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, dot(A, W_out, A))
    return NormalWeightedMeanPrecision(dot(A, ξ_out), W)
end

# if A is a scalar, then the input is either univariate or multivariate
@rule typeof(*)(:in, Marginalisation) (m_out::F, m_A::PointMass{ <: Real }, meta::AbstractCorrection) where { F <: NormalDistributionsFamily } = begin
    A = mean(m_A)
    ξ_out, W_out = weightedmean_precision(m_out)
    W = correction!(meta, A^2 * W_out)
    return convert(promote_variate_type(F, NormalWeightedMeanPrecision), A * ξ_out, W)
end

# specialized versions for mean-covariance parameterization
@rule typeof(*)(:in, Marginalisation) (m_out::MvNormalMeanCovariance, m_A::PointMass{ <: AbstractMatrix }, meta::AbstractCorrection) = begin
    A = mean(m_A)
    μ_out, Σ_out = mean_cov(m_out)

    z = fastcholesky(Σ_out)
    tmp = A' / z
    W = correction!(meta, tmp * A)

    return MvNormalWeightedMeanPrecision(tmp * μ_out, W)
end

@rule typeof(*)(:in, Marginalisation) (m_out::MvNormalMeanCovariance, m_A::PointMass{ <: AbstractVector }, meta::AbstractCorrection) = begin
    A = mean(m_A)
    μ_out, Σ_out = mean_cov(m_out)

    z = fastcholesky(Σ_out)
    tmp = A' / z
    W = correction!(meta, dot(tmp, A))
    
    return NormalWeightedMeanPrecision(dot(tmp, μ_out), W)
end