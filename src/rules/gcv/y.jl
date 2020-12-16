export rule

@rule GCV(:y, Marginalisation) (m_x::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin
    γ_1 = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
    γ_2 = exp(-mean(q_κ) * mean(q_z) + 0.5 * γ_1)
    γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))

    return NormalMeanVariance(mean(m_x), var(m_x) + 1.0 / (γ_2 * γ_3))
end