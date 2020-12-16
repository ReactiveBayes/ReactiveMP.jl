export marginalrule

@marginalrule Beta(:out_a_b) (m_out::Beta, m_a::Dirac, m_b::Dirac) = begin
    return (out = prod(ProdPreserveParametrisation(), Beta(mean(m_a), mean(m_b)), m_out), a = m_a, b = m_b)
end