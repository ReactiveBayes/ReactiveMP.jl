export rule

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass, m_in::PointMass) = PointMass(mean(m_A) * mean(m_in))

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass, m_in::F) where { F <: NormalDistributionsFamily } = begin
    A = mean(m_A)
    return convert(promote_variate_type(F, NormalMeanVariance), A * mean(m_in), A * cov(m_in) * A')
end