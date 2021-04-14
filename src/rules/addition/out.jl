export rule

@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::PointMass) = PointMass(mean(m_in1) + mean(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanVariance, m_in2::PointMass) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::NormalMeanVariance) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanVariance, m_in2::NormalMeanVariance) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in2))

@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanPrecision, m_in2::NormalMeanPrecision) = begin
    p1, p2 = precision(m_in1), precision(m_in2)
    return NormalMeanPrecision(mean(m_in1) + mean(m_in2), p1 * p2 / (p1 + p2))
end

@rule typeof(+)(:out, Marginalisation) (m_in1::ComplexNormal, m_in2::ComplexNormal) = ComplexNormal(mean(m_in1) - mean(m_in2), var(m_in1) + var(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::ComplexNormal) = ComplexNormal(mean(m_in1) - mean(m_in2), var(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::ComplexNormal, m_in2::PointMass) = ComplexNormal(mean(m_in1) - mean(m_in2), var(m_in1))