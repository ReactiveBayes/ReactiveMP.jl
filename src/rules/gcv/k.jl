export rule

@rule GCV(:κ, Marginalisation) (q_y_x::Any, q_z::Any, q_ω::Any, meta::GCVMetadata) = begin
    
    y_x_mean, y_x_var = mean_cov(q_y_x)
    ω_mean, ω_var     = mean_var(q_ω)
    z_mean, z_var     = mean_var(q_z)

    A = exp(-ω_mean + ω_var / 2)
    psi = (y_x_mean[1] - y_x_mean[2]) ^ 2 + y_x_var[1, 1] + y_x_var[2, 2] - y_x_var[1, 2] - y_x_var[2, 1]

    a = z_mean
    b = psi * A
    c = -a
    d = z_var

    return ExponentialLinearQuadratic(get_approximation(meta), a, b, c, d)
end

@rule GCV(:κ, Marginalisation) (q_y::Any, q_x::Any, q_z::Any, q_ω::Any, meta::GCVMetadata) = begin
    
    y_mean, y_var = mean_var(q_y)
    x_mean, x_var = mean_var(q_x)
    ω_mean, ω_var = mean_var(q_ω)
    z_mean, z_var = mean_var(q_z)

    A = exp(-ω_mean + ω_var / 2)
    psi = (y_mean - x_mean) ^ 2 + y_var + x_var

    a = z_mean
    b = psi * A
    c = -a
    d = z_var

    return ExponentialLinearQuadratic(get_approximation(meta), a, b, c, d)
end