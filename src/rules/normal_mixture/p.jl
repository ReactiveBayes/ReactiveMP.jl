export rule

@rule NormalMixture((:p, k), Marginalisation) (q_out::Any, q_switch::Any, q_m::UnivariateNormalDistributionsFamily) = begin
    m_mean_k, v_mean_k = mean_cov(q_m)
    m_out, v_out       = mean_cov(q_out)
    z_bar = probvec(q_switch)
    return GammaShapeRate(one(eltype(z_bar)) + z_bar[k] / 2, z_bar[k] * (v_out + v_mean_k + abs2(m_out - m_mean_k)) / 2)
end

@rule NormalMixture((:p, k), Marginalisation) (q_out::Any, q_switch::Any, q_m::MultivariateNormalDistributionsFamily) = begin
    m_mean_k, v_mean_k = mean_cov(q_m)
    m_out, v_out       = mean_cov(q_out)
    z_bar = probvec(q_switch)
    d = length(m_mean_k)
    return Wishart(one(eltype(z_bar)) + z_bar[k] + d, cholinv(z_bar[k]*( v_out + v_mean_k + (m_out - m_mean_k)*(m_out - m_mean_k)')))
end