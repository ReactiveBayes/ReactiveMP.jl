
# Outbound messages toward the previous state `x`. Delegates to the AR rules with the effective
# (q_θ, q_γ) marginals derived from the joint q(w) = MvNormalGamma.
@rule ConjugateAR(:x, Marginalisation) (
    m_y::NormalDistributionsFamily, q_w::MvNormalGamma, meta::ARMeta
) = begin
    q_θ, q_γ = conjugatear_effective_marginals(q_w)
    return @call_rule AR(:x, Marginalisation) (m_y = m_y, q_θ = q_θ, q_γ = q_γ, meta = meta)
end

@rule ConjugateAR(:x, Marginalisation) (
    q_y::Any, q_w::MvNormalGamma, meta::ARMeta
) = begin
    q_θ, q_γ = conjugatear_effective_marginals(q_w)
    return @call_rule AR(:x, Marginalisation) (q_y = q_y, q_θ = q_θ, q_γ = q_γ, meta = meta)
end
