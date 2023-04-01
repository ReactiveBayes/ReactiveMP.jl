export softdot, SoftDot

import StatsFuns: log2π

struct SoftDot end
const softdot = SoftDot

@node softdot Stochastic [y, (θ, aliases = [theta]), x, (γ, aliases = [gamma])]

# TODO: Generalize q_γ for GammaDistributionsFamily by comparing derivations for GammaShapeRate
@average_energy softdot (q_y::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_x::NormalDistributionsFamily, q_γ::GammaShapeRate) = begin
    m_y, V_y = mean_cov(q_y)
    m_θ, V_θ = mean_cov(q_θ)
    m_x, V_x = mean_cov(q_x)
    m_γ = mean(q_γ)
    AE = 0
    AE -= mean(log, q_γ)
    AE += log2π
    AE += m_γ * (V_y + m_y^2 - 2m_γ*m_y*m_θ'm_x + mul_trace(V_θ,V_x) + m_x'V_θ*m_x + m_θ'*(V_x + m_x*m_x')*m_θ)
    AE /= 2
    return AE
end
