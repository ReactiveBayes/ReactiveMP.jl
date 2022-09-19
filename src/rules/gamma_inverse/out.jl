export rule

# case: inbound messages on edges [α, β]
@rule GammaInverse(:out, Marginalisation) (m_α::PointMass, m_β::PointMass) = GammaInverse(mean(m_α), mean(m_β))

# case: inbound variational posterior on edges [α, β]
@rule GammaInverse(:out, Marginalisation) (q_α::PointMass, q_β::PointMass) = GammaInverse(mean(q_α), mean(q_β))
