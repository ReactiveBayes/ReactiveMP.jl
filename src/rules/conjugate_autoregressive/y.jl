
# Outbound messages toward the next state `y`. The ConjugateAR factor is identical to AR; with
# the joint q(w) = MvNormalGamma the required moments map to effective (q_θ, q_γ) marginals
# (see `conjugatear_effective_marginals`), so we delegate to the tested AR rules.
@rule ConjugateAR(:y, Marginalisation) (
    m_x::NormalDistributionsFamily, q_w::MvNormalGamma, meta::ARMeta
) = begin
    q_θ, q_γ = conjugatear_effective_marginals(q_w)
    return @call_rule AR(:y, Marginalisation) (
        m_x = m_x, q_θ = q_θ, q_γ = q_γ, meta = meta
    )
end

@rule ConjugateAR(:y, Marginalisation) (
    q_x::Any, q_w::MvNormalGamma, meta::ARMeta
) = begin
    q_θ, q_γ = conjugatear_effective_marginals(q_w)
    return @call_rule AR(:y, Marginalisation) (
        q_x = q_x, q_θ = q_θ, q_γ = q_γ, meta = meta
    )
end
