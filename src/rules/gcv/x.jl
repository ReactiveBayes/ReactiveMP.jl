export rule

@rule GCV(:x, Marginalisation) (
    m_y::UniNormalOrExpLinQuad,
    q_z::Any,
    q_κ::Any,
    q_ω::Any,
    meta::Union{<:GCVMetadata, Nothing},
) = begin
    y_mean, y_var = mean_var(m_y)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    # Effective noise variance is `1/⟨e^{-(κz+ω)}⟩`. Computed from the summed log-exponents
    # rather than `inv(A * B)`, which yields `NaN` when one factor overflows while the other
    # underflows even though their product is representable. See `__gcv_log_noise_precision`.
    log_ab = __gcv_log_noise_precision(q_z, q_κ, q_ω)

    return NormalMeanVariance(y_mean, y_var + exp(-log_ab))
end

@rule GCV(:x, Marginalisation) (
    q_y::Any, q_z::Any, q_κ::Any, q_ω::Any, meta::Union{<:GCVMetadata, Nothing}
) = begin
    y_mean = mean(q_y)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    # See `__gcv_log_noise_precision`.
    log_ab = __gcv_log_noise_precision(q_z, q_κ, q_ω)

    return NormalMeanVariance(y_mean, exp(-log_ab))
end
