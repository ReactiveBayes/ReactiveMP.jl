function compute_z_and_D(my, Vy, mx, Vx, mW, order, ds, Fs, es)
    D = sum(sum(es[j]' * mW * es[i] * Fs[i]' * (mx * mx' + Vx) * Fs[j] for i in 1:ds) for j in 1:ds)
    z = sum(Fs[i]' * ((mx * mx' + Vx') * mar_shift(order, ds)' + mx * my') * mW * es[i] for i in 1:ds)

    return z, D
end

@rule MAR(:a, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) = begin
    order, ds = getorder(meta), getdimensionality(meta)
    Fs, es    = getmasks(meta), getunits(meta)
    dim       = order * ds

    m, V = mean_cov(q_y_x)
    F = Multivariate

    my, Vy = ar_slice(F, m, 1:dim), ar_slice(F, V, 1:dim, 1:dim)
    mx, Vx = ar_slice(F, m, (dim + 1):(2dim)), ar_slice(F, V, (dim + 1):(2dim), (dim + 1):(2dim))

    mΛ = mean(q_Λ)
    mW = mar_transition(order, mΛ)

    z, D = compute_z_and_D(my, Vy, mx, Vx, mW, order, ds, Fs, es)
    return MvNormalWeightedMeanPrecision(z, D)
end

@rule MAR(:a, Marginalisation) (q_y::MultivariateNormalDistributionsFamily, q_x::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) = begin
    order, ds = getorder(meta), getdimensionality(meta)
    F = Multivariate

    dim = order * ds

    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    mΛ     = mean(q_Λ)

    mW = mar_transition(order, mΛ)

    z, D = compute_z_and_D(my, Vy, mx, Vx, mW, order, ds, getmasks(meta), getunits(meta))
    return MvNormalWeightedMeanPrecision(z, D)
end
