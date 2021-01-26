export rule

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass, m_in::PointMass) = PointMass(mean(m_in1) * mean(m_in2))

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass, m_in::NormalMeanVariance) = begin
    A = mean(m_A)
    return NormalMeanVariance(A * mean(m_in), A * cov(m_in) * A')
end

@rule typeof(*)(:out, Marginalisation) (m_A::PointMass, m_in::MvNormalMeanCovariance) = begin
    A = mean(m_A)
    return MvNormalMeanCovariance(A * mean(m_in), A * cov(m_in) * A')
end