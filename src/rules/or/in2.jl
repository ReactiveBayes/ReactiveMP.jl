@rule OR(:in2, Marginalisation) (m_out::Bernoulli, m_in1::Bernoulli) = begin
    return @call_rule typeof(OR)(:in1, Marginalisation) (m_out = m_out, m_in2 = m_in1, meta = meta)
end
