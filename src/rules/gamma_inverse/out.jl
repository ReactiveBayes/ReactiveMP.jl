export rule

# case: inbound messages on edges [α, β] sum-product == belief propagation
@rule GammaInverse(:out, Marginalisation) (m_α::Any, m_β::Any) = GammaInverse(mean(m_α), mean(m_β))

# case: inbound variational posterior on edges [α, β]
@rule GammaInverse(:out, Marginalisation) (q_α::Any, q_β::Any) = GammaInverse(mean(q_α), mean(q_β))
