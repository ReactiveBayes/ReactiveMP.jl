export marginalrule

@marginalrule typeof(*)(:A_in) (m_out::NormalDistributionsFamily, m_A::PointMass, m_in::F) where { F <: NormalDistributionsFamily } = begin
    # A = mean(m_A)
    # xi_out, W_out = weightedmean_precision(m_out)
    # W = A' * W_out * A
    # b_in = convert(promote_variate_type(F, NormalWeightedMeanPrecision), A' * xi_out, W)  
    b_in = @call_rule typeof(*)(:in, Marginalisation) (m_out = m_out, m_A = m_A, meta = meta)  
    q_in = prod(ProdAnalytical(), b_in, m_in)
    return (A = m_A, in = q_in)
end