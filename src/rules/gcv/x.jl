export rule

@rule GCV(:x, Marginalisation) (m_y::UniNormalOrExpLinQuad, q_z::Any, q_κ::Any, q_ω::Any) = begin
    y_mean, y_var = mean_var(m_y)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    ksi = κ_mean^2 * z_var + z_mean^2 * κ_var + z_var * κ_var
    A = exp(-ω_mean + ω_var / 2)
    B = exp(-κ_mean * z_mean + ksi / 2)

    return NormalMeanVariance(y_mean, y_var + inv(A * B))
end

@rule GCV(:x, Marginalisation) (q_y::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin
    y_mean        = mean(q_y)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    ksi = κ_mean^2 * z_var + z_mean^2 * κ_var + z_var * κ_var
    A = exp(-ω_mean + ω_var / 2)
    B = exp(-κ_mean * z_mean + ksi / 2)

    return NormalMeanVariance(y_mean, inv(A * B))
end
