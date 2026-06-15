
@rule MvNormalGamma(:out, Marginalisation) (
    m_μ::PointMass, m_Λ::PointMass, m_α::PointMass, m_β::PointMass
) = MvNormalGamma(mean(m_μ), mean(m_Λ), mean(m_α), mean(m_β))

@rule MvNormalGamma(:out, Marginalisation) (
    q_μ::Any, q_Λ::Any, q_α::Any, q_β::Any
) = MvNormalGamma(mean(q_μ), mean(q_Λ), mean(q_α), mean(q_β))
