
@marginalrule GCV(:y_x) (
    m_y::UniNormalOrExpLinQuad,
    m_x::UniNormalOrExpLinQuad,
    q_z::Any,
    q_κ::Any,
    q_ω::Any,
    meta::Union{<:GCVMetadata, Nothing},
) = begin
    y_mean, y_precision = mean_precision(m_y)
    x_mean, x_precision = mean_precision(m_x)

    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    # See `__gcv_log_noise_precision`: the coupling is `⟨e^{-(κz+ω)}⟩`, computed from summed
    # log-exponents so it cannot come out `NaN` for individually extreme factors.
    ab = exp(__gcv_log_noise_precision(q_z, q_κ, q_ω))
    W = [y_precision+ab -ab; -ab x_precision+ab]
    ξ = [y_mean * y_precision; x_mean * x_precision]

    return MvNormalWeightedMeanPrecision(ξ, W)
end
