export rule

@rule GCV(:y, Marginalisation) (m_x::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin

    x_mean, x_var = mean_var(m_x)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    ksi = κ_mean ^ 2 * z_var + z_mean ^ 2 * κ_var + z_var * κ_var
    A = exp(-ω_mean + ω_var / 2)
    B = exp(-κ_mean * z_mean + ksi / 2)

    return NormalMeanVariance(x_mean, x_var + inv(A * B))
end

@rule GCV(:y, Marginalisation) (q_x::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin

    x_mean        = mean(q_x)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    ksi = κ_mean ^ 2 * z_var + z_mean ^ 2 * κ_var + z_var * κ_var
    A = exp(-ω_mean + ω_var / 2)
    B = exp(-κ_mean * z_mean + ksi / 2)

    return NormalMeanVariance(x_mean, inv(A * B))
end