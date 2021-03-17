export rule

@rule AR(:γ, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_θ::NormalDistributionsFamily, meta::ARMeta) = begin
    order = getorder(meta)
    F     = getvform(meta)

    myx, Vyx = mean(q_y_x), cov(q_y_x)

    mA, Vθ = as_companion_matrix(mean(q_θ)), cov(q_θ)
    my, Vy = ar_slice(F, myx, 1:order), ar_slice(F, Vyx, 1:order, 1:order)
    mx, Vx = ar_slice(F, myx, order + 1:2order), ar_slice(F, Vyx, order+1:2order, order+1:2order)
    Vyx    = ar_slice(F, Vyx, order+1:2order, 1:order)

    B = (Vy + my * my')[1, 1] - 2 * (mA * (Vyx + mx * my'))[1, 1] + (mA * (Vx + mx * mx') * mA')[1, 1] + tr(Vθ * (Vx + mx * mx'))

    return GammaShapeRate(3.0 / 2.0, B / 2.0)
end
