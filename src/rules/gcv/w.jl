export rule


@rule GCV(:ω, Marginalisation) (q_y_x::Any, q_z::Any, q_κ::Any, meta::GCVMetadata) = begin
    
    m, v = mean(q_y_x), cov(q_y_x)

    γ = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
    A = exp(-mean(q_κ) * mean(q_z) + γ / 2)
    psi = (m[1] - m[2]) ^ 2 + v[1, 1] + v[2, 2] - v[1, 2] - v[2, 1]

    a = one(typeof(γ))
    b = psi * A
    c = -one(typeof(γ))
    d = zero(typeof(γ))

    return ExponentialLinearQuadratic(get_approximation(meta),a, b, c, d)
end

@rule GCV(:ω, Marginalisation) (q_y::Any, q_x::Any, q_z::Any, q_κ::Any, meta::GCVMetadata) = begin
    
    my, vy = mean(q_y), cov(q_y)
    mx, vx = mean(q_x), cov(q_x)

    γ = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
    A = exp(-mean(q_κ) * mean(q_z) + γ / 2)
    psi = (my - mx) ^ 2 + vy + vx

    a = mean(q_z)
    b = psi * A
    c = -a
    d = var(q_z)

    return ExponentialLinearQuadratic(get_approximation(meta),a, b, c, d)
end