export rule

@rule typeof(*)(:out, Marginalisation) (m_A::Dirac, m_in::Dirac) = Dirac(mean(m_in1) * mean(m_in2))

@rule typeof(*)(:out, Marginalisation) (m_A::Dirac, m_in::NormalMeanVariance) = begin
    A = mean(m_A)
    return NormalMeanVariance(A * mean(m_in), A * cov(m_in) * A')
end

@rule typeof(*)(:out, Marginalisation) (m_A::Dirac, m_in::MvNormalMeanCovariance) = begin
    A = mean(m_A)
    return MvNormalMeanCovariance(A * mean(m_in), A * cov(m_in) * A')
end