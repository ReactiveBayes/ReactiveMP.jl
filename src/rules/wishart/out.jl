
# Belief Propagation                #
# --------------------------------- #
@rule Wishart(:out, Marginalisation) (m_ν::PointMass, m_S::PointMass) = WishartMessage(mean(m_ν), cholinv(mean(m_S)))

# Variational                       # 
# --------------------------------- #
@rule Wishart(:out, Marginalisation) (q_ν::Any, m_S::PointMass) = WishartMessage(mean(q_ν), cholinv(mean(m_S)))
@rule Wishart(:out, Marginalisation) (m_ν::PointMass, q_S::Any) = WishartMessage(mean(m_ν), cholinv(mean(q_S)))
@rule Wishart(:out, Marginalisation) (q_ν::Any, q_S::Any)       = WishartMessage(mean(q_ν), cholinv(mean(q_S)))
