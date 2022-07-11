# Belief Propagation                #
# --------------------------------- #
@rule InverseWishart(:out, Marginalisation) (m_ν::PointMass, m_S::PointMass) = InverseWishartMessage(mean(m_ν), mean(m_S))

# Variational                       # 
# --------------------------------- #
@rule InverseWishart(:out, Marginalisation) (q_ν::Any, m_S::PointMass) = InverseWishartMessage(mean(q_ν), mean(m_S))
@rule InverseWishart(:out, Marginalisation) (m_ν::PointMass, q_S::Any) = InverseWishartMessage(mean(m_ν), mean(q_S))
@rule InverseWishart(:out, Marginalisation) (q_ν::Any, q_S::Any)       = InverseWishartMessage(mean(q_ν), mean(q_S))
