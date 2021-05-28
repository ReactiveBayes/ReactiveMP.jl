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

@rule typeof(+)(:in2, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in1::MultivariateNormalDistributionsFamily) = begin
    mout, vout = mean_cov(m_out)
    m1, v1 = mean_cov(m_in1)
    return MvNormalMeanCovariance(mout- m1, vout + v1)
end