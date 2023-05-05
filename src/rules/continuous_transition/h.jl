@rule ContinuousTransition(:h, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::CTMeta) = begin
    dy, dx = getdimensionality(meta)
    Fs, es = getmasks(meta), getunits(meta)

    myx, Vyx = mean_cov(q_y_x)

    mx, Vx = myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = myx[1:dy], Vyx[1:dy, 1:dy]
    Vyx    = Vyx[1:dy, (dy + 1):end]

    mΛ = mean(q_Λ)

    D = sum(sum(es[i]' * mΛ * es[j] * Fs[i]' * (mx * mx' + Vx) * Fs[j] for i in 1:length(Fs)) for j in 1:length(Fs))
    z = sum(Fs[i]' * (mx * my' + Vyx') * mΛ * es[i] for i in 1:length(Fs))

    return MvNormalWeightedMeanPrecision(z, D)
end
