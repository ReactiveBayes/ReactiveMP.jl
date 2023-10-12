
@marginalrule Categorical(:out_p) (m_out::Categorical, m_p::PointMass) = begin
    return convert_paramfloattype((out = prod(ClosedProd(), Categorical(mean(m_p)), m_out), p = m_p))
end

@marginalrule Categorical(:out_p) (m_out::PointMass, m_p::Dirichlet) = begin
    p = prod(ClosedProd(), Dirichlet(probvec(m_out) .+ one(eltype(probvec(m_out)))), m_p)
    return convert_paramfloattype((out = m_out, p = p))
end
