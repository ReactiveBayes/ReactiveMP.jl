using StatsFuns: logistic
@rule sigmoid(:out, Marginalisation) (q_in::UnivariateNormalDistributionsFamily,q_ξ::PointMass) = begin
    m_in = mean(q_in)
    p = logistic(m_in)
    return Categorical(p, 1 - p)
end

