export rule

@rule typeof(*)(:in, Marginalisation) (m_out::PointMass, m_A::PointMass) = PointMass(mean(m_in1) * mean(m_in2))

@rule typeof(*)(:in, Marginalisation) (m_out::NormalMeanVariance, m_A::PointMass) = begin
    A_inv = cholinv(mean(m_A))
    return NormalMeanVariance(A_inv * mean(m_out), A_inv * cov(m_out) * A_inv')
end

@rule typeof(*)(:in, Marginalisation) (m_out::MvNormalMeanCovariance, m_A::PointMass) = begin
    A_inv = cholinv(mean(m_A))
    return MvNormalMeanCovariance(A_inv * mean(m_out), A_inv * cov(m_out) * A_inv')
end