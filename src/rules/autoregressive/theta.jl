export rule

@rule AR(:θ, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta{F}) where F = begin
    order = meta.order

    myx, Vyx = mean(q_y_x), cov(q_y_x)

    my, Vy = myx[1:order], Vyx[1:order,1:order]
    mx, Vx = myx[order+1:end], Vyx[order+1:end, order+1:end]
    Vxy    = Vyx[order+1:end,1:order]

    mγ = mean(q_γ)

    D = mγ * (Vx + mx * mx')
    c = zeros(order); c[1] = 1.0
    mθ, Vθ = cholinv(D) * (Vxy + mx * my') * mγ * c, cholinv(D)

    return convert(promote_variate_type(F, NormalMeanVariance), mθ, Vθ)
end
