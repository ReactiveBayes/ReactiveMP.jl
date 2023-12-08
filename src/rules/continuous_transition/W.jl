function compute_delta(my, Vy, mx, Vx, Vyx, mA, Va, ma, Fs, es)
    G₁ = (my * my' + Vy)
    G₂ = ((my * mx' + Vyx) * mA')
    G₃ = transpose(G₂)
    Ex_xx = mx * mx' + Vx
    G₅ = sum(sum(es[i] * ma' * Fs[i]'Ex_xx * Fs[j] * ma * es[j]' for i in 1:length(Fs)) for j in 1:length(Fs))
    G₆ = sum(sum(es[i] * tr(Fs[i]' * Ex_xx * Fs[j] * Va) * es[j]' for i in 1:length(Fs)) for j in 1:length(Fs))
    return G₁ - G₂ - G₃ + G₅ + G₆
end

@rule ContinuousTransition(:W, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, meta::CTMeta) = begin
    dy, dx = getdimensionality(meta)

    ma, Va = mean_cov(q_a)
    Fs, es = getjacobians(meta, ma), getunits(meta)

    mA = ctcompanion_matrix(ma, sqrt.(var(q_a)), meta)
    myx, Vyx = mean_cov(q_y_x)

    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = @views myx[1:dy], Vyx[1:dy, 1:dy]
    Vyx    = @views Vyx[1:dy, (dy + 1):end]

    Δ = compute_delta(my, Vy, mx, Vx, Vyx, mA, Va, ma, Fs, es)

    return WishartFast(dy + 2, Δ)
end
