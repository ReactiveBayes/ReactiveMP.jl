import StatsFuns: log2π

@node MvNormalWeightedMeanPrecision Stochastic [ out, (ξ, aliases = [xi, weightedmean]), (Λ, aliases = [invcov, precision]) ]

@average_energy MvNormalWeightedMeanPrecision (q_out::Any, q_ξ::Any, q_Λ::Any) = begin
    m_mean, v_mean = mean_cov(q_ξ)
    m_out, v_out   = mean_cov(q_out)
    return (
        ndims(q_out) * log2π - chollogdet(mean(q_Λ)) +
        tr(mean(q_Λ) * (v_out + v_mean + (m_out - m_mean) * (m_out - m_mean)'))
    ) / 2
end

@average_energy MvNormalWeightedMeanPrecision (q_out_ξ::Any, q_Λ::Any) = begin
    m, V = mean_cov(q_out_ξ)
    d = div(ndims(q_out_ξ), 2)
    return @views (
        d * log2π + -chollogdet(mean(q_Λ)) +
        tr(
            mean(q_Λ) * (
                V[1:d, 1:d] - V[1:d, d+1:end] - V[d+1:end, 1:d] + V[d+1:end, d+1:end] +
                (m[1:d] - m[d+1:end]) * (m[1:d] - m[d+1:end])'
            )
        )
    ) / 2
end
