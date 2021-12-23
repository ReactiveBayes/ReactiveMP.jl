
@rule typeof(*)(:out, Marginalisation) (m_A::PointMass, m_in::PointMass) = PointMass(mean(m_A) * mean(m_in))

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass{ <: Real }, m_in::GammaDistributionsFamily) = begin
    return GammaShapeRate(shape(m_in), rate(m_in) / mean(m_A))
end

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass{ <: AbstractMatrix }, m_in::F) where { F <: NormalDistributionsFamily } = begin
    A = mean(m_A)
    μ_in, Σ_in = mean_cov(m_in)
    return convert(promote_variate_type(F, NormalMeanVariance), A * μ_in, A * Σ_in * A')
end

#------------------------
# AbstractVector * UnivariateNormalDistributions
#------------------------
# We consider the following updates as a special case of the MatrixVariate * Multivariate updates.
# Namely, Ax = y, where A ∈ R^{nx1}, x ∈ R^1, and y ∈ R^n. In this case, the matrix A
# can be represented by a n-dimensional vector, and x by a scalar. Before computation,
# quantities are converted to their proper dimensions (see situational sketch below).
#
#     | a ~ AbstractVector -> R^{nx1}
#     v  out ~ Multivariate -> R^n
# -->[x]-->
# in1 ~ Univariate -> R^1
@rule typeof(*)(:out, Marginalisation) (m_A::PointMass{ <: AbstractVector }, m_in::UnivariateNormalDistributionsFamily) = begin
    a = mean(m_A)

    μ_in, v_in = mean_var(m_in)

    # TODO: check, do we need correction! here? (ForneyLab does not have any correction in this case)
    # TODO: Σ in this rule is guaranteed to be ill-defined, has rank equal to one and has determinant equal to zero
    μ = μ_in * a
    Σ = mul_inplace!(v_in, a * a')

    return MvNormalMeanCovariance(μ, Σ)
end