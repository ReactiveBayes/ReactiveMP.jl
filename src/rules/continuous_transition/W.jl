function compute_delta(my, Vy, mx, Vx, Vyx, mA, Va, ma, Fs)
    dy = length(my)
    G₁ = (my * my' + Vy)
    G₂ = ((my * mx' + Vyx) * mA')
    G₃ = transpose(G₂)
    Ex_xx = rank1update(Vx, mx)
    G₅ = zeros(eltype(ma), dy, dy)
    G₆ = zeros(eltype(ma), dy, dy)
    mamat = ma * ma'
    for (i, j) in Iterators.product(1:dy, 1:dy)
        tmp = Fs[i]' * Ex_xx * Fs[j]
        G₅[i, j] = tr(tmp * mamat)
        G₆[i, j] = tr(tmp * Va)
    end
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
