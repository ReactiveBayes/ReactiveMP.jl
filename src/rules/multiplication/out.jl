export rule

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass, m_in::PointMass) = PointMass(mean(m_A) * mean(m_in))

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass, m_in::F) where { F <: NormalDistributionsFamily } = begin
    A = mean(m_A)
    μ_in, Σ_in = mean_cov(m_in)
    return convert(promote_variate_type(F, NormalMeanVariance), A * μ_in, A * Σ_in * A')
end