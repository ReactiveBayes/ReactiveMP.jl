export rule

@rule AR(:γ, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_θ::NormalDistributionsFamily, meta::ARMeta) = begin
    order = meta.order

    myx, Vyx = mean(q_y_x), cov(q_y_x)

    mA, Vθ = as_CMatrix(mean(q_θ)), cov(q_θ)
    my, Vy = myx[1:order], Vyx[1:order,1:order]
    mx, Vx = myx[order + 1:end], Vyx[order+1:end, order+1:end]
    Vyx = Vyx[order+1:end,1:order]

    B = (Vy + my*my')[1, 1] - 2*(mA*(Vyx + mx*my'))[1, 1] + (mA*(Vx + mx*mx')*mA')[1, 1] + tr(Vθ*(Vx + mx*mx'))

    return GammaShapeRate(3.0 / 2.0, B / 2.0)
end
