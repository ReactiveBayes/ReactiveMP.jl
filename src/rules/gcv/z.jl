export rule

@rule GCV(:z, Marginalisation) (q_y_x::Any, q_κ::Any, q_ω::Any, meta::GCVMetadata) = begin

    y_x_mean, y_x_v = mean_cov(q_y_x)
    κ_mean, κ_var   = mean_var(q_κ)
    ω_mean, ω_var   = mean_var(q_ω)

    psi = @inbounds (y_x_mean[1] - y_x_mean[2]) ^ 2 + y_x_v[1,1] + y_x_v[2,2] - y_x_v[1,2] - y_x_v[2,1]
    A = exp(-ω_mean + ω_var / 2)

    a = κ_mean
    b = psi * A
    c = -κ_mean
    d = κ_var

    return ExponentialLinearQuadratic(get_approximation(meta), a, b, c, d)
end

@rule GCV(:z, Marginalisation) (q_y::Any, q_x::Any, q_κ::Any, q_ω::Any, meta::GCVMetadata) = begin

    y_mean, y_var = mean_var(q_y)
    x_mean, x_var = mean_var(q_x)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    psi = (y_mean - x_mean) ^ 2 + y_var+ x_var
    A = exp(-ω_mean + ω_var / 2)

    a = κ_mean
    b = psi * A
    c = -κ_mean
    d = κ_var

    return ExponentialLinearQuadratic(get_approximation(meta), a, b, c, d)
end