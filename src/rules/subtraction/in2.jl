@rule typeof(-)(:in2, Marginalisation) (m_out::Any, m_in1::Any, meta::Any) = begin
    @call_rule typeof(+)(:in2, Marginalisation) (m_out = m_in1, m_in1 = m_out, meta = meta)
end
