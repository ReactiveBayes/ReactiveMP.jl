
@rule AR(:γ, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_θ::NormalDistributionsFamily, meta::ARMeta) = begin
    order = getorder(meta)
    F     = getvform(meta)

    y_x_mean, y_x_cov = mean_cov(q_y_x)
    θ_mean, θ_cov = mean_cov(q_θ)

    mA, Vθ = as_companion_matrix(θ_mean), θ_cov
    my, Vy = ar_slice(F, y_x_mean, 1:order), ar_slice(F, y_x_cov, 1:order, 1:order)
    mx, Vx = ar_slice(F, y_x_mean, (order + 1):2order), ar_slice(F, y_x_cov, (order + 1):2order, (order + 1):2order)
    Vyx    = ar_slice(F, y_x_cov, (order + 1):2order, 1:order)

    C = rank1update(Vx, mx)
    R = rank1update(Vy, my)
    L = rank1update(Vyx, mx, my)

    B = first(R) - 2 * first(mA * L) + first(mA * C * mA') + mul_trace(Vθ, C)

    return GammaShapeRate(convert(eltype(B), 3//2), B / 2)
end
