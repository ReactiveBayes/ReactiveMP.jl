
import Distributions: MatrixNormal

@rule MatrixNormal(:out, Marginalisation) (m_M::PointMass, m_U::PointMass, m_V::PointMass) = MatrixNormal(
    mean(m_M), mean(m_U), mean(m_V)
)
