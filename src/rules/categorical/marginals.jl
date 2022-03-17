
@marginalrule Categorical(:out_p) (m_out::Categorical, m_p::PointMass) = begin
    return (out = prod(ProdAnalytical(), Categorical(mean(m_p)), m_out), p = m_p)
end