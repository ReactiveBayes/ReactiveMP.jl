export softdot, SoftDot

import StatsFuns: log2π
"""
    SoftDot
    
The SoftDot node can be used as a substitute for the dot product operator delta node (the outgoing variable is the dot product of two others). 
It softens the delta constraint by adding a Gaussian noise as follows:

    y ~ N(dot(θ, x), γ^(-1))

Interfaces:
1. y - result of the "soft" dot product,
2. θ - first variable to be multiplied,
3. x - second variable to be multiplied,
4. γ - precision of the Gaussian noise.

The advantage of using SoftDot is that it offers tractable and optimized closed-form variational messages 
for both Belief Propagation and Variational Message Passing.

See also: [`softdot`](@ref)
"""
struct SoftDot end

"""
Alias for the `SoftDot` node.
"""
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

    order    = length(mθ)
    F        = order == 1 ? Univariate : Multivariate
    mx, Vx   = ar_slice(F, myx, (2):(order + 1)), ar_slice(F, Vyx, (2):(order + 1), (2):(order + 1))
    my1, Vy1 = first(myx), first(Vyx)
    Vy1x     = ar_slice(F, Vyx, 1, (2):(order + 1))

    # Equivalent to AE = (-mean(log, q_γ) + log2π + mγ*(Vy1+my1^2 - 2*mθ'*(Vy1x + mx*my1) + tr(Vθ*Vx) + mx'*Vθ*mx + mθ'*(Vx + mx*mx')*mθ)) / 2
    AE = (-mean(log, q_γ) + log2π + mγ * (Vy1 + my1^2 - 2 * mθ' * (Vy1x + mx * my1) + mul_trace(Vθ, Vx) + dot(mx, Vθ, mx) + dot(mθ, Vx, mθ) + abs2(dot(mθ, mx)))) / 2

    return AE
end
