function compute_delta(my, Vy, mx, Vx, Vyx, mH, Vh, mh, Fs, es)
    G₁ = (my * my' + Vy)
    G₂ = ((my * mx' + Vyx) * mH')
    G₃ = transpose(G₂)
    Ex_xx = mx * mx' + Vx
    G₅ = sum(sum(es[i] * mh' * Fs[i]'Ex_xx * Fs[j] * mh * es[j]' for i in 1:length(Fs)) for j in 1:length(Fs))
    G₆ = sum(sum(es[i] * tr(Fs[i]' * Ex_xx * Fs[j] * Vh) * es[j]' for i in 1:length(Fs)) for j in 1:length(Fs))
    return G₁ - G₂ - G₃ + G₅ + G₆
end

@rule ContinuousTransition(:Λ, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_h::MultivariateNormalDistributionsFamily, meta::CTMeta) = begin
    dy, dx = getdimensionality(meta)
    Fs, es = getmasks(meta), getunits(meta)

    mh, Vh = mean_cov(q_h)
    mH = ctcompanion_matrix(mh, meta)
    myx, Vyx = mean_cov(q_y_x)

    mx, Vx = myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = myx[1:dy], Vyx[1:dy, 1:dy]
    Vyx    = Vyx[1:dy, (dy + 1):end]

    Δ = compute_delta(my, Vy, mx, Vx, Vyx, mH, Vh, mh, Fs, es)

    # NOTE: WishartMessage stores inverse of scale matrix
    return WishartMessage(dy + 2, Δ)
end
