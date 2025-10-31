function compute_delta(my, Vy, mx, Vx, Vyx, mA, Va, ma, Fs)
    dy = length(my)
    G₁ = (my * my' + Vy)

    G₂ = ((my * mx' + Vyx) * mA')
    G₃ = transpose(G₂)
    Ex_xx = rank1update(Vx, mx)
    G₅ = zeros(eltype(ma), dy, dy)
    G₆ = zeros(eltype(ma), dy, dy)
    mamat = ma * ma'

    Y = similar(Ex_xx)
    Z = similar(Ex_xx)

    @inbounds for (i, j) in Iterators.product(1:dy, 1:dy)
        mul!(Y, Ex_xx, Fs[j])
        mul!(Z, Fs[i]', Y)

        G₅[i, j] = mul_trace(Z, mamat)
        G₆[i, j] = mul_trace(Z, Va)

        # tmp = Fs[i]' * Ex_xx * Fs[j]
        # G₅[i, j] = mul_trace(tmp, mamat)
        # G₆[i, j] = mul_trace(tmp, Va)
    end

    G = G₁ - (G₂ + G₃) .+ Symmetric(G₅ + G₆)

    return G
end

# VMP: Stuctured
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

# VMP: Mean-field
@rule ContinuousTransition(:W, Marginalisation) (q_y::Any, q_x::Any, q_a::Any, meta::CTMeta) = begin
    ma, Va = mean_cov(q_a)
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)

    Fs = getjacobians(meta, ma)
    dy = length(Fs)

    epsilon = sqrt.(var(q_a))
    mA = ctcompanion_matrix(ma, epsilon, meta)

    Δ = compute_delta(my, Vy, mx, Vx, zeros(eltype(ma), dy, length(mx)), mA, Va, ma, Fs)

    return WishartFast(dy + 2, Δ)
end
