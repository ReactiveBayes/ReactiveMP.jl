export marginalrule

@marginalrule GCV(:y_x) (m_y::Any, m_x::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin

    ksi = mean(q_κ) ^ 2 * var(q_z) + mean(q_z) ^ 2 * var(q_κ) + var(q_z) * var(q_κ)
    A = exp(-mean(q_ω) + var(q_ω) / 2)
    B = exp(-mean(q_κ) * mean(q_z) + ksi / 2)
    W = [ precision(m_y) + A * B -A * B; -A * B precision(m_x) + A * B ]
    m = cholinv(W) * [ mean(m_y) * precision(m_y); mean(m_x) * precision(m_x) ]

    return MvNormalMeanPrecision(m, W)
end