
@marginalrule Categorical(:out_p) (m_out::Categorical, m_p::PointMass) = begin
    return (out = prod(ProdAnalytical(), Categorical(mean(m_p)), m_out), p = m_p)
end

@marginalrule Categorical(:out_p) (m_out::PointMass, m_p::Dirichlet) = begin
    p = prod(ProdAnalytical(), Dirichlet(probvec(m_out) .+ one(eltype(probvec(m_out)))), m_p)
    return (out = m_out, p = p)
end