
@rule AR(:γ, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_θ::Any, meta::ARMeta) = begin
    order = getorder(meta)
    F     = getvform(meta)

    y_x_mean, y_x_cov = mean_cov(q_y_x)
    mθ, vθ = mean_cov(q_θ)

    mA, Vθ = as_companion_matrix(mθ), vθ
    my, Vy = ar_slice(F, y_x_mean, 1:order), ar_slice(F, y_x_cov, 1:order, 1:order)
    mx, Vx = ar_slice(F, y_x_mean, (order + 1):(2order)), ar_slice(F, y_x_cov, (order + 1):(2order), (order + 1):(2order))
    Vyx = ar_slice(F, y_x_cov, (order + 1):(2order), 1:order)

    C = rank1update(Vx, mx)
    R = rank1update(Vy, my)
    L = rank1update(Vyx, mx, my)

    B = first(R) - 2 * first(mA * L) + first(mA * C * mA') + mul_trace(Vθ, C)

    return GammaShapeRate(convert(eltype(B), 3//2), B / 2)
end

@rule AR(:γ, Marginalisation) (q_y::Any, q_x::Any, q_θ::Any, meta::ARMeta) = begin
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    mθ, Vθ = mean_cov(q_θ)

    B = first(Vy) + first(my)^2 - 2 * first(my) * mθ' * mx + mx' * Vθ * mx + mθ' * (Vx + mx * mx') * mθ

    return GammaShapeRate(convert(eltype(B), 3//2), B / 2)
end
