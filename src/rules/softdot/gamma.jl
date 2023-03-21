# Belief propagation: does not exist for softdot

# Variational MP
@rule softdot(:γ, Marginalisation) (q_y::NormalDistributionsFamily, q_x::MultivariateNormalDistributionsFamily, q_θ::MultivariateNormalDistributionsFamily) = begin
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    mθ, Vθ = mean_cov(q_θ)
    α = 1.5
    β = 0.5 * (Vy + my * transpose(my))
    β -= my*transpose(mθ)*mx
    β += 0.5 * ( mul_trace(Vx, Vθ) + transpose(mθ) * Vx * mθ + transpose(mx) * Vθ * mx + transpose(mθ) * mx * transpose(mx) * mθ )
    return GammaShapeRate(α, β)
end
