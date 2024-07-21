# Belief propagation: does not exist for softdot

# Variational MP: Mean-field
@rule softdot(:γ, Marginalisation) (q_y::Any, q_θ::Any, q_x::Any) = begin
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    mθ, Vθ = mean_cov(q_θ)
    T = promote_paramfloattype(q_y, q_x, q_θ)
    α = convert(T, 3 / 2)
    β = (Vy + my * my') / 2
    β -= my * mθ' * mx
    β += (mul_trace(Vx, Vθ) + mθ'Vx * mθ + mx'Vθ * mx + mθ'mx * mx'mθ) / 2
    return GammaShapeRate(α, β)
end

# Variational MP: Structured
@rule softdot(:γ, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_θ::Any) = begin
    # q_y is always Univariate
    order = length(q_y_x) - 1
    F     = order == 1 ? Univariate : Multivariate

    y_x_mean, y_x_cov = mean_cov(q_y_x)
    mθ, Vθ = mean_cov(q_θ)

    my, Vy = first(y_x_mean), first(y_x_cov)
    mx, Vx = ar_slice(F, y_x_mean, 2:(order + 1)), ar_slice(F, y_x_cov, 2:(order + 1), 2:(order + 1))
    Vyx = ar_slice(F, y_x_cov, 2:(order + 1))

    C = rank1update(Vx, mx)
    R = rank1update(Vy, my)
    L = Vyx + mx * my

    B = first(R) - 2 * first(mθ' * L) + first(mθ' * C * mθ) + mul_trace(Vθ, C)

    return GammaShapeRate(convert(eltype(B), 3//2), B / 2)
end
