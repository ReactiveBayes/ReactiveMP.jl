@rule typeof(+)(:in2, Marginalisation) (m_out::Any, m_in1::Any, meta::Any) = begin
    return @call_rule typeof(+)(:in1, Marginalisation) (m_out = m_out, m_in2 = m_in1, meta = meta)
end