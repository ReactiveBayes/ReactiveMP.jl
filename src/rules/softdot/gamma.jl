# Belief propagation: does not exist for softdot

# Variational MP: Mean-field
@rule softdot(:γ, Marginalisation) (q_y::Any, q_θ::Any, q_x::Any) = begin
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    mθ, Vθ = mean_cov(q_θ)
    T = promote_samplefloattype(q_y, q_x, q_θ)
    α = convert(T, 3 / 2)
    β = (Vy + my * my') / 2
    β -= my * mθ' * mx
    β += (mul_trace(Vx, Vθ) + mθ'Vx * mθ + mx'Vθ * mx + mθ'mx * mx'mθ) / 2
    return GammaShapeRate(α, β)
end
