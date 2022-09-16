
@marginalrule GCV(:y_x) (
    m_y::UniNormalOrExpLinQuad,
    m_x::UniNormalOrExpLinQuad,
    q_z::Any,
    q_κ::Any,
    q_ω::Any,
    meta::Union{<:GCVMetadata, Nothing}
) = begin
    y_mean, y_precision = mean_precision(m_y)
    x_mean, x_precision = mean_precision(m_x)

    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    ksi = κ_mean^2 * z_var + z_mean^2 * κ_var + z_var * κ_var
    A = exp(-ω_mean + ω_var / 2)
    B = exp(-κ_mean * z_mean + ksi / 2)
    W = [y_precision+A*B -A*B; -A*B x_precision+A*B]
    ξ = [y_mean * y_precision; x_mean * x_precision]

    return MvNormalWeightedMeanPrecision(ξ, W)
end
