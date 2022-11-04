export rule

@rule NormalMixture((:p, k), Marginalisation) (q_out::Any, q_switch::Any, q_m::Any) = begin
    m_mean_k, v_mean_k = mean_cov(q_m)
    m_out, v_out       = mean_cov(q_out)
    z_bar              = probvec(q_switch)

    return rule_nm_p_k(variate_form(q_out), m_mean_k, v_mean_k, m_out, v_out, z_bar, k)
end

function rule_nm_p_k(::Type{Univariate}, m_mean_k, v_mean_k, m_out, v_out, z_bar, k)
    return GammaShapeRate(one(eltype(z_bar)) + z_bar[k] / 2, z_bar[k] * (v_out + v_mean_k + abs2(m_out - m_mean_k)) / 2)
end

function rule_nm_p_k(::Type{Multivariate}, m_mean_k, v_mean_k, m_out, v_out, z_bar, k)
    return WishartMessage(one(eltype(z_bar)) + z_bar[k] + length(m_mean_k), z_bar[k] * (v_out + v_mean_k + (m_out - m_mean_k) * (m_out - m_mean_k)'))
end
