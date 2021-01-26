export rule

@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::PointMass) = PointMass(mean(m_out) - mean(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::NormalMeanVariance) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanVariance, m_in1::PointMass) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanVariance, m_in1::NormalMeanVariance) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))

@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::NormalMeanPrecision) = NormalMeanPrecision(mean(m_out) - mean(m_in1), precision(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::NormalMeanPrecision) = begin
    p1, p2 = precision(m_out), precision(m_in1)
    return NormalMeanPrecision(mean(m_out) - mean(m_in1), p1 * p2 / (p1 + p2))
end
