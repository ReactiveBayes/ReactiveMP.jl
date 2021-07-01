
@marginalrule typeof(+)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    return (in1 = prod(ProdAnalytical(), NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out)), m_in1), in2 = m_in2)
end

@marginalrule typeof(+)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_in1, W_in1 = weightedmean_precision(m_in1)
    xi_in2, W_in2 = weightedmean_precision(m_in2)
    
    return MvNormalWeightedMeanPrecision([ xi_in1 + xi_out; xi_in2 + xi_out ], [ W_in1 + W_out W_out; W_out W_in2 + W_out])
end