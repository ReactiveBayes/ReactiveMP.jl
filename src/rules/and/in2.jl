@rule typeof(AND)(:in2, Marginalisation) (m_out::Bernoulli, m_in1::Bernoulli, meta::Any) = begin
    return @call_rule typeof(AND)(:in1, Marginalisation) (m_out = m_out, m_in2 = m_in1, meta = meta)
end
