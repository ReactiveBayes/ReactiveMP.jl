@rule Sigmoid(:ξ, Marginalisation) (q_out::Any, q_in::UnivariateNormalDistributionsFamily) = begin
    m_in, v_in = mean_cov(q_in)
    return sqrt(m_in^2 + v_in)
end