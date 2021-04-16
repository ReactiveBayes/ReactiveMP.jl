export rule

@rule GCV(:z, Marginalisation) (q_y_x::Any, q_κ::Any, q_ω::Any, meta::GCVMetadata) = begin

    m, v = mean(q_y_x), cov(q_y_x)
    psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]
    A = exp(-mean(q_ω)+var(q_ω) / 2)

    a = mean(q_κ)
    b = psi * A
    c = -mean(q_κ)
    d = var(q_κ)

    return ExponentialLinearQuadratic(get_approximation(meta), a, b, c, d)
end