export rule

@rule AR(:θ, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta{F}) where F = begin
    order = meta.order

    myx, Vyx = mean(q_y_x), cov(q_y_x)

    my, Vy = arslice(F, myx, 1:order), arslice(F, Vyx, 1:order, 1:order)
    mx, Vx = arslice(F, myx, order+1:2order), arslice(F, Vyx, order+1:2order, order+1:2order)
    Vyx    = arslice(F, Vyx, order+1:2order, 1:order)

    mγ = mean(q_γ)

    D = mγ * (Vx + mx * mx')
    c = uvector(F, order)

    mθ, Vθ = cholinv(D) * (Vyx + mx * my') * mγ * c, cholinv(D)
    
    return convert(promote_variate_type(F, NormalMeanVariance), mθ, Vθ)
end
