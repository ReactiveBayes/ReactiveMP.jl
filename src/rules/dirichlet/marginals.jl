export marginalrule

@marginalrule Dirichlet(:out_a) (m_out::Dirichlet, m_a::Dirac) = begin
    return (out = prod(ProdPreserveParametrisation(), Dirichlet(mean(m_a)), m_out), a = m_a)
end