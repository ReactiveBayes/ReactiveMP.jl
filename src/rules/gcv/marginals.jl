@marginalrule(
    form       => Type{ GCV }, 
    on         => :y_x,
    messages   => (m_y::Any, m_x::Any),
    marginals  => (q_z::Any, q_κ::Any, q_ω::Any),
    meta       => Nothing,
    begin 
        γ_1 = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
        γ_2 = exp(-mean(q_κ) * mean(q_z) + 0.5 * γ_1)
        γ_3 = exp(-mean(q_ω) + 0.5 * var(q_ω))
        γ23 = γ_2 * γ_3

        Λ = cholinv([ (precision(m_y) + γ23) -γ23; -γ23 (precision(m_x) + γ23) ])
        m = Λ * [ mean(m_y) * precision(m_y); mean(m_x) * precision(m_x) ]

        return MvNormalMeanCovariance(m, Λ)
    end
)