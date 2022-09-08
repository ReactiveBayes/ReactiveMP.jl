export rule

# Belief propagation
# case: inbound messages on edges [α, θ] sum-product
@rule GammaInverse(:out, Marginalisation) (m_α::PointMass, m_θ::PointMass) = GammaInverse(mean(m_α), mean(m_θ))

# Variational MP
# case: inbound variational posterior on edges [α, θ]
@rule GammaInverse(:out, Marginalisation) (q_α::Any, q_θ::Any) = GammaInverse(mean(q_α), mean(q_θ))
