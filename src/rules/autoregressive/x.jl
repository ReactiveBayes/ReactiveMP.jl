
@rule AR(:x, Marginalisation) (m_y::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::Any, meta::ARMeta) =
    begin
        mθ, Vθ = mean_cov(q_θ)
        my, Vy = mean_cov(m_y)

        mγ = mean(q_γ)

        mA = as_companion_matrix(mθ)
        mV = ar_transition(getvform(meta), getorder(meta), mγ)

        C = mA' * inv(add_transition(Vy, mV))

        W = C * mA + mγ * Vθ
        ξ = C * my

        return convert(promote_variate_type(getvform(meta), NormalWeightedMeanPrecision), ξ, W)
    end

@rule AR(:x, Marginalisation) (q_y::Any, q_θ::Any, q_γ::Any, meta::ARMeta) = begin
    mθ, Vθ = mean_cov(q_θ)
    mγ = mean(q_γ)

    mA = as_companion_matrix(mθ)
    mV = ar_transition(getvform(meta), getorder(meta), mγ)

    C = mA' * pinv(mV)
    W = C * mA + mγ * Vθ
    ξ = C * mean(q_y)
    return convert(promote_variate_type(getvform(meta), NormalWeightedMeanPrecision), ξ, W)
end
