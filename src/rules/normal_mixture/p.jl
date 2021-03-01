export rule

@rule NormalMixture((:p, k), Marginalisation) (q_out::Any, q_switch::Any, q_m::UnivariateNormalDistributionsFamily) = begin
    m_mean_k, v_mean_k = mean(q_m), cov(q_m)
    m_out, v_out = mean(q_out), cov(q_out)
    z_bar = probvec(q_switch)
    return GammaShapeRate(1.0 + 0.5 * z_bar[k], 0.5 * z_bar[k] * (v_out + v_mean_k + abs2(m_out - m_mean_k)))
end

@rule NormalMixture((:p, k), Marginalisation) (q_out::Any, q_switch::Any, q_m::MultivariateNormalDistributionsFamily) = begin
    m_mean_k, v_mean_k = mean(q_m), cov(q_m)
    m_out, v_out       = mean(q_out), cov(q_out)
    z_bar = probvec(q_switch)
    d = length(m_mean_k)
    return Wishart(1.0 + z_bar[k] + d, cholinv(z_bar[k]*( v_out + v_mean_k + (m_out - m_mean_k)*(m_out - m_mean_k)')))
end