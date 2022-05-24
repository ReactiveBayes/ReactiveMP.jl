
@rule AR(:θ, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_γ::Any, meta::ARMeta) = begin
    order = getorder(meta)
    F     = getvform(meta)

    myx, Vyx = mean_cov(q_y_x)
    my, Vy   = ar_slice(F, myx, 1:order), ar_slice(F, Vyx, 1:order, 1:order)
    mx, Vx   = ar_slice(F, myx, (order+1):2order), ar_slice(F, Vyx, (order+1):2order, (order+1):2order)
    Vyx      = ar_slice(F, Vyx, (order+1):2order, 1:order)

    mγ = mean(q_γ)

    # Equivalent to W = mγ * (Vx + mx * mx')
    W = rank1update(Vx, mx)
    W = mul_inplace!(mγ, W)

    c = ar_unit(getvform(meta), order)

    # Equivalent to ξ = (Vyx + mx * my') * c * mγ
    ξ = rank1update(Vyx, mx, my) * c
    ξ = mul_inplace!(mγ, ξ)

    return convert(promote_variate_type(F, NormalWeightedMeanPrecision), ξ, W)
end

@rule AR(:θ, Marginalisation) (q_y::Any, q_x::Any, q_γ::Any, meta::ARMeta) = begin
    order = getorder(meta)

    mx, Vx = mean_cov(q_x)

    my, mγ = mean(q_y), mean(q_γ)

    mV = ar_transition(getvform(meta), getorder(meta), mγ)
    c = ar_unit(getvform(meta), order)

    ξ = mx * c' * pinv(mV) * my
    W = mγ * (Vx + mx * mx')
    return convert(promote_variate_type(getvform(meta), NormalWeightedMeanPrecision), ξ, W)
end
