export rule

@rule typeof(dot)(:out, Marginalisation) (m_in1::PointMass, m_in2::MvNormalMeanPrecision) = NormalMeanPrecision(mean(m_in1)'*mean(m_in2), inv(mean(m_in1)'*var(m_in2)*mean(m_in1)))
@rule typeof(dot)(:out, Marginalisation) (m_in1::MvNormalMeanPrecision, m_in2::PointMass) = NormalMeanPrecision(mean(m_in2)'*mean(m_in1), inv(mean(m_in2)'*var(m_in1)*mean(m_in2)))
