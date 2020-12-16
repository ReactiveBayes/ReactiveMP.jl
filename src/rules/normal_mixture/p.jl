export rule

@rule NormalMixture((:p, k), Marginalisation) (q_out::Any, q_switch::Any, q_m::NTuple{N1, NormalMeanVariance}, q_p::NTuple{N2, Gamma}) where { N1, N2 } = begin
    m_mean_k, v_mean_k = mean(q_m[k]), cov(q_m[k])
    m_out, v_out = mean(q_out), cov(q_out)
    z_bar = probvec(q_switch)
    return Gamma(1.0 + 0.5 * z_bar[k], inv(0.5 * z_bar[k] * (v_out + v_mean_k + abs2(m_out - m_mean_k))))
end