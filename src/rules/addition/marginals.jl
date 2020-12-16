export marginalrule

@marginalrule typeof(+)(:in1_in2) (m_out::NormalMeanVariance, m_in1::NormalMeanVariance, m_in2::Dirac) = begin
    (in1 = prod(ProdPreserveParametrisation(), NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out)), m_in1), in2 = m_in2)
end