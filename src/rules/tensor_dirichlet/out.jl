
@rule TensorDirichlet(:out, Marginalisation) (m_a::PointMass,) = TensorDirichlet(mean(m_a))

@rule TensorDirichlet(:out, Marginalisation) (q_a::PointMass,) = TensorDirichlet(mean(q_a))
