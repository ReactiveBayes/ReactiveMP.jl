
@rule MatrixDirichlet(:out, Marginalisation) (m_a::PointMass, ) = begin
    return MatrixDirichlet(mean(m_a))
end