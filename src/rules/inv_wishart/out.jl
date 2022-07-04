# Belief Propagation                #
# --------------------------------- #
@rule InvWishart(:out, Marginalisation) (m_ν::PointMass, m_S::PointMass) = InvWishart(mean(m_ν), mean(m_S))

# Variational                       # 
# --------------------------------- #
@rule InvWishart(:out, Marginalisation) (q_ν::Any, m_S::PointMass) = InvWishart(mean(q_ν), mean(m_S))
@rule InvWishart(:out, Marginalisation) (m_ν::PointMass, q_S::Any) = InvWishart(mean(m_ν), mean(q_S))
@rule InvWishart(:out, Marginalisation) (q_ν::Any, q_S::Any)       = InvWishart(mean(q_ν), mean(q_S))
