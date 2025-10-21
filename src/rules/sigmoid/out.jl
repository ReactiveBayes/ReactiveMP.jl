using StatsFuns: logistic
@rule Sigmoid(:out, Marginalisation) (q_in::UnivariateNormalDistributionsFamily, q_ζ::PointMass) = begin
    m_in = mean(q_in)
    ζ_hat = mean(q_ζ)
    p = logistic(m_in)
    T = promote_type(eltype(m_in), eltype(ζ_hat))
    probs = clamp.([p, 1 - p], tiny, 1 - tiny)
    probs ./= sum(probs)
    probs_T = convert(Vector{T}, probs)
    return Categorical(probs_T)
end
