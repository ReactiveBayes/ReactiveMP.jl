export rule

@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::PointMass) = PointMass(mean(m_out) - mean(m_in2))

@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanVariance, m_in2::PointMass) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::NormalMeanVariance) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_in2))

@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::PointMass) = NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::NormalMeanPrecision) = NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::NormalMeanPrecision) = begin
    p1, p2 = precision(m_out), precision(m_in2)
    return NormalMeanPrecision(mean(m_out) - mean(m_in2), p1 * p2 / (p1 + p2))
end

@rule typeof(+)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = begin
    mout, vout = mean_cov(m_out)
    m2, v2 = mean_cov(m_in2)
    return MvNormalMeanCovariance(mout- m2, vout + v2)
end

@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    mout = mean(m_out)
    m2, v2 = mean_cov(m_in2)
    return MvNormalMeanCovariance(mout- m2, v2)
end
