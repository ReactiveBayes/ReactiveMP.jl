function compute_delta(my, Vy, mx, Vx, Vyx, mA, Va, ma, Fs)
    dy = length(my)
    G₁ = (my * my' + Vy)
    G₂ = ((my * mx' + Vyx) * mA')
    G₃ = transpose(G₂)
    Ex_xx = mx * mx' + Vx
    G₅ = sum(sum(StandardBasisVector(dy, i) * ma' * Fs[i]'Ex_xx * Fs[j] * ma * StandardBasisVector(dy, j)' for i in 1:dy) for j in 1:dy)
    G₆ = sum(sum(StandardBasisVector(dy, i) * tr(Fs[i]' * Ex_xx * Fs[j] * Va) * StandardBasisVector(dy, j)' for i in 1:dy) for j in 1:dy)
    return G₁ - G₂ - G₃ + G₅ + G₆
end

@rule ContinuousTransition(:W, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, meta::CTMeta) = begin
    ma, Va = mean_cov(q_a)
    Fs = getjacobians(meta, ma)
    dy = length(Fs)

    epsilon = sqrt.(var(q_a))
    mA = ctcompanion_matrix(ma, epsilon, meta)

    myx, Vyx = mean_cov(q_y_x)

    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = @views myx[1:dy], Vyx[1:dy, 1:dy]
    Vyx    = @views Vyx[1:dy, (dy + 1):end]

    Δ = compute_delta(my, Vy, mx, Vx, Vyx, mA, Va, ma, Fs)

    return WishartFast(dy + 2, Δ)
end
