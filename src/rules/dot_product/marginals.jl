export marginalrule

@marginalrule typeof(dot)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass{ <: AbstractVector }, m_in2::MultivariateNormalDistributionsFamily) = begin
    ξ, W = weightedmean_precision(m_out)
    x = mean(m_in1)
    q_in2 = prod(ProdAnalytical(), m_in2, MvNormalWeightedMeanPrecision(x * ξ, x * W * x'))
    return (in1 = m_in1, in2 = q_in2)
end

@marginalrule typeof(dot)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass{ <: AbstractVector })= begin
    ξ, W = weightedmean_precision(m_out)
    x = mean(m_in2)
    q_in1 = prod(ProdAnalytical(), m_in1, MvNormalWeightedMeanPrecision(x * ξ, x * W * x'))
    return (in1 = q_in1, in2 = m_in2)
end