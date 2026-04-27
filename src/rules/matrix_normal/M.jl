
import Distributions: MatrixNormal

@rule MatrixNormal(:M, Marginalisation) (m_out::PointMass, m_U::PointMass, m_V::PointMass) = MatrixNormal(
    mean(m_out), mean(m_U), mean(m_V)
)
