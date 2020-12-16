export marginalrule

@marginalrule Bernoulli(:out_p) (m_out::Dirac, m_p::Beta) = begin
    return (out = m_out, p = prod(ProdPreserveParametrisation(), Beta(1.0 + mean(m_out), 2.0 - mean(m_out)), m_p))
end