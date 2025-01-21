
@marginalrule TensorDirichlet(:out_a) (m_out::TensorDirichlet, m_a::PointMass) = begin
    return convert_paramfloattype((out = prod(ClosedProd(), TensorDirichlet(mean(m_a)), m_out), a = m_a))
end
