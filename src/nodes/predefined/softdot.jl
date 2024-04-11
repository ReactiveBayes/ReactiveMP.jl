export softdot, SoftDot

import StatsFuns: log2π

struct SoftDot end
const softdot = SoftDot

@node softdot Stochastic [y, (θ, aliases = [theta]), x, (γ, aliases = [gamma])]

@average_energy softdot (q_y::Any, q_θ::Any, q_x::Any, q_γ::Any) = begin
    m_y, V_y = mean_cov(q_y)
    m_θ, V_θ = mean_cov(q_θ)
    m_x, V_x = mean_cov(q_x)
    m_γ = mean(q_γ)
    return (-mean(log, q_γ) + log2π + m_γ * (V_y + m_y^2 - 2m_γ * m_y * m_θ'm_x + mul_trace(V_θ, V_x) + m_x'V_θ * m_x + m_θ' * (V_x + m_x * m_x') * m_θ)) / 2
end
