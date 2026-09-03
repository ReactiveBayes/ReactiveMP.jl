@marginalrule GaussianCoupling(:out_in) (
    m_out::UnivariateNormalDistributionsFamily,
    m_in::UnivariateNormalDistributionsFamily,
    q_a::PointMass,
) = begin
    # q(out, in) ∝ m_out(out) m_in(in) exp(a⋅out⋅in)
    ξ = [weightedmean(m_out), weightedmean(m_in)]
    W = [precision(m_out) -mean(q_a); -mean(q_a) precision(m_in)]
    # → MvNormalWeightedMeanPrecision(ξ, W)   (proper iff precision(m_out)⋅precision(m_in) > mean(q_a)²)
    return MvNormalWeightedMeanPrecision(ξ, W)
end
