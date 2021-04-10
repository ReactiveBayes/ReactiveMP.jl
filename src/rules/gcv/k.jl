export rule

@rule GCV(:κ, Marginalisation) (q_y_x::Any, q_z::Any, q_ω::Any,meta::Any) = begin
    
    m, v = mean(q_y_x), cov(q_y_x)

    A = exp(-mean(q_ω) + 0.5 * var(q_ω))
    psi = (m[1] - m[2]) ^ 2 + v[1, 1] + v[2, 2] - v[1, 2] - v[2, 1]

    a = mean(q_z)
    b = psi * A
    c = -a
    d = var(q_z)

    return ExponentialLinearQuadratic(get_variance_approximation(GCV, meta),a, b, c, d)
end

