
@marginalrule MatrixNormal(:out_M_U_V) (
    m_out::PointMass, m_M::PointMass, m_U::PointMass, m_V::PointMass
) = convert_paramfloattype((out = m_out, M = m_M, U = m_U, V = m_V))
