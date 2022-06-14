
@rule MatrixDirichlet(:out, Marginalisation) (m_a::PointMass,) = begin
    return MatrixDirichlet(mean(m_a))
end

@rule MatrixDirichlet(:out, Marginalisation) (q_a::PointMass,) = begin
    return MatrixDirichlet(mean(q_a))
end
