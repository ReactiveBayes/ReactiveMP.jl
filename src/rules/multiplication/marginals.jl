export marginalrule

@marginalrule typeof(*)(:A_in) (m_out::NormalDistributionsFamily, m_A::PointMass, m_in::F) where { F <: NormalDistributionsFamily } = begin
    A = mean(m_A)
    W = A' * precision(m_out) * A
    b_in = convert(promote_variate_type(F, NormalWeightedMeanPrecision), A' * weightedmean(m_out), W)    
    q_in = prod(ProdBestSuitableParametrisation(), b_in, m_in)
    return (A = m_A, in = q_in)
end