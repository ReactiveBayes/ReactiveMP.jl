export rule


@rule GCV(:ω, Marginalisation) (q_y_x::Any, q_z::Any, q_κ::Any, meta::GCVMetadata) = begin

    y_x_mean, y_x_var = mean_cov(q_y_x)
    z_mean, z_var     = mean_var(q_z)
    κ_mean, κ_var     = mean_var(q_κ)

    γ = z_mean ^ 2 * κ_var + κ_mean ^ 2 * z_var + z_var * κ_var
    A = exp(-κ_mean * z_mean + γ / 2)
    psi = (y_x_mean[1] - y_x_mean[2]) ^ 2 + y_x_var[1, 1] + y_x_var[2, 2] - y_x_var[1, 2] - y_x_var[2, 1]

    a = one(typeof(γ))
    b = psi * A
    c = -one(typeof(γ))
    d = zero(typeof(γ))

    return ExponentialLinearQuadratic(get_approximation(meta), a, b, c, d)
end

@rule GCV(:ω, Marginalisation) (q_y::Any, q_x::Any, q_z::Any, q_κ::Any, meta::GCVMetadata) = begin
    
    y_mean, y_var = mean_var(q_y)
    x_mean, x_var = mean_var(q_x)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)

    γ = z_mean ^ 2 * κ_var + κ_mean ^ 2 * z_var + z_var * κ_var
    A = exp(-κ_mean * z_mean + γ / 2)
    psi = (y_mean - x_mean) ^ 2 + y_var + x_var

    a = z_mean
    b = psi * A
    c = -a
    d = z_var

    return ExponentialLinearQuadratic(get_approximation(meta), a, b, c, d)
end