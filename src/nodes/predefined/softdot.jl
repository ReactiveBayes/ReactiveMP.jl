export softdot, SoftDot

import StatsFuns: log2π

"""
SoftDot [aliases = [Softdot]]: ReactiveMP node for message passing. 

The SoftDot node can be used as a substitution for the mutiplication operator delta node (the outgoing variable is the multiplcation of two others). It soften the delta constraint by adding a Gaussian noise as follow:

y ~ N( x * θ' , γ^(-1))

Interfaces:
1. y - result of the soft multiplication
2. θ - first variable to be multiplied
3. x - second variable to be multiplied
4. γ - precision of the gaussian noise

Delta node for multiplication implies intractable messages. 
SoftDot implementation offers tractable closed-form variational messages for diverce constraints (naive mean field, struturec mean field, moment matching...).
It is  recommanded to use variational message as belief progagation for SoftDot also leads to untractable messages.
"""
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

@average_energy softdot (q_y_x::MultivariateNormalDistributionsFamily, q_θ::Any, q_γ::Any) = begin
    mθ, Vθ   = mean_cov(q_θ)
    myx, Vyx = mean_cov(q_y_x)
    mγ       = mean(q_γ)

    order = length(mθ)
    F     = order == 1 ? Univariate : Multivariate

    mx, Vx   = ar_slice(F, myx, (order + 1):(2order)), ar_slice(F, Vyx, (order + 1):(2order), (order + 1):(2order))
    my1, Vy1 = first(myx), first(Vyx)
    Vy1x     = ar_slice(F, Vyx, 1, (order + 1):(2order))

    # Equivalent to AE = (-mean(log, q_γ) + log2π + mγ*(Vy1+my1^2 - 2*mθ'*(Vy1x + mx*my1) + tr(Vθ*Vx) + mx'*Vθ*mx + mθ'*(Vx + mx*mx')*mθ)) / 2
    AE = (-mean(log, q_γ) + log2π + mγ * (Vy1 + my1^2 - 2 * mθ' * (Vy1x + mx * my1) + mul_trace(Vθ, Vx) + dot(mx, Vθ, mx) + dot(mθ, Vx, mθ) + abs2(dot(mθ, mx)))) / 2

    return AE
end
