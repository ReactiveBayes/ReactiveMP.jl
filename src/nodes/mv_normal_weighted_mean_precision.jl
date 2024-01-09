import StatsFuns: log2π

@node MvNormalWeightedMeanPrecision Stochastic [out, (ξ, aliases = [xi, weightedmean]), (Λ, aliases = [invcov, precision])]

@average_energy MvNormalWeightedMeanPrecision (q_out::Any, q_ξ::Any, q_Λ::Any) = begin
    m_ξ, v_ξ     = mean_cov(q_ξ)
    m_out, v_out = mean_cov(q_out)
    return (ndims(q_out) * log2π - mean(chollogdet, q_Λ) + tr(mean(q_Λ) * (m_out * m_out' + v_out)) - 2*dot(m_out,m_ξ) + tr(mean(cholinv, q_Λ) * (m_ξ * m_ξ' + v_ξ))) / 2
end
