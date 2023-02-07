
@rule MAR(:a, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) = begin
    order, ds = getorder(meta), getdimensionality(meta)
    F = Multivariate

    dim = order * ds

    m, V = mean_cov(q_y_x)

    my, Vy = ar_slice(F, m, 1:dim), ar_slice(F, V, 1:dim, 1:dim)
    mx, Vx = ar_slice(F, m, (dim + 1):(2dim)), ar_slice(F, V, (dim + 1):(2dim), (dim + 1):(2dim))
    Vyx    = ar_slice(F, V, 1:dim, (dim + 1):(2dim))

    mΛ = mean(q_Λ)
    mW = mar_transition(order, mΛ)

    # this should be inside MARMeta
    es = [uvector(dim, i) for i in 1:ds]
    Fs = [mask_mar(order, ds, i) for i in 1:ds]
    S = mar_shift(order, ds)

    # NOTE: prove that sum(Fs[i]'*((mx*mx'+Vx')*S')*mW*es[i] for i in 1:ds) == 0.0
    D = sum(sum(es[i]' * mW * es[j] * Fs[i]' * (mx * mx' + Vx) * Fs[j] for i in 1:ds) for j in 1:ds)
    z = sum(Fs[i]' * (mx * my' + Vyx') * mW * es[i] for i in 1:ds)

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
    S  = mar_shift(order, ds)

    # this should be inside MARMeta
    es = [uvector(dim, i) for i in 1:ds]
    Fs = [mask_mar(order, ds, i) for i in 1:ds]

    D = sum(sum(es[j]' * mW * es[i] * Fs[i]' * (mx * mx' + Vx) * Fs[j] for i in 1:ds) for j in 1:ds)
    z = sum(Fs[i]' * ((mx * mx' + Vx') * S' + mx * my') * mW * es[i] for i in 1:ds)

    return MvNormalWeightedMeanPrecision(z, D)
end
