export marginalrule

@marginalrule typeof(dot)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass{ <: AbstractVector }, m_in2::MultivariateNormalDistributionsFamily) = begin
    m, V = mean(m_out), cov(m_out)
    x = mean(m_in1)
    q_in2 = prod(ProdAnalytical(), m_in2, MvNormalWeightedMeanPrecision(x * weightedmean(m_out), x * precision(m_out) * x'))
    return (in1 = m_in1, in2 = q_in2)
end

@marginalrule typeof(dot)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass, m_in2::UnivariateNormalDistributionsFamily) = begin
    m, V = mean(m_out), cov(m_out)
    x = mean(m_in1)
    q_in2 = prod(ProdAnalytical(), m_in2, NormalWeightedMeanPrecision(x * weightedmean(m_out), x * precision(m_out) * x'))
    return (in1 = m_in1, in2 = q_in2)
end

@marginalrule typeof(dot)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass{ <: AbstractVector }) = begin
    m, V = mean(m_out), cov(m_out)
    x = mean(m_in2)
    q_in2 = prod(ProdAnalytical(), m_in1, MvNormalWeightedMeanPrecision(x * weightedmean(m_out), x * precision(m_out) * x'))
    return (in1 = m_in2, in2 = q_in2)
end

@marginalrule typeof(dot)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    m, V = mean(m_out), cov(m_out)
    x = mean(m_in2)
    q_in2 = prod(ProdAnalytical(), m_in1, NormalWeightedMeanPrecision(x * weightedmean(m_out), x * precision(m_out) * x'))
    return (in1 = m_in2, in2 = q_in2)
end
