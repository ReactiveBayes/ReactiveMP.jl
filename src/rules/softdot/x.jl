# Belief propagation: does not exist for softdot

# Variational MP: Mean-field
@rule softdot(:x, Marginalisation) (q_y::Any, q_θ::Any, q_γ::Any) = begin
    my = mean(q_y)
    mθ, Vθ = mean_cov(q_θ)
    mγ = mean(q_γ)
    Dx = mγ * (Vθ + mθ * mθ')
    zx = mγ * mθ * my
    return convert(promote_variate_type(variate_form(typeof(q_θ)), NormalWeightedMeanPrecision), zx, Dx)
end

# Variational MP: Structured
@rule softdot(:x, Marginalisation) (m_y::UnivariateNormalDistributionsFamily, q_θ::Any, q_γ::Any) = begin
    # the naive call of AR rule is not possible, because the softdot rule expects m_y to be a UnivariateNormalDistributionsFamily
    mθ, Vθ = mean_cov(q_θ)
    my, Vy = mean_cov(m_y)

    mγ = mean(q_γ)

    mV = inv(mγ)

    C = mθ * inv(add_transition(Vy, mV))

    W = C * mθ' + mγ * Vθ
    ξ = C * my

    return convert(promote_variate_type(variate_form(typeof(q_θ)), NormalWeightedMeanPrecision), ξ, W)
end
