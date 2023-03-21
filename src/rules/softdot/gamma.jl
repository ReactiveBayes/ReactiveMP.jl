# Belief propagation: does not exist for softdot

# Variational MP
@rule softdot(:γ, Marginalisation) (q_y::Any, q_x::Any, q_θ::Any) = begin
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    mθ, Vθ = mean_cov(q_θ)
    α = 1.5
    β = 0.5 * (Vy + my*my')
    β -= my*mθ'*mx
    β += 0.5 * ( mul_trace(Vx, Vθ) + mθ'Vx*mθ + mx'Vθ*mx + mθ'mx*mx'mθ )
    return GammaShapeRate(α, β)
end
